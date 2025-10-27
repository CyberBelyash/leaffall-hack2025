import numpy as np
from haversine import haversine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import onnxruntime as ort
import os
import requests
from fastapi.middleware.cors import CORSMiddleware
from itertools import permutations
import math

# Путь к ONNX-модели
ONNX_PATH = "gat3_policy_model.onnx"

if not os.path.exists(ONNX_PATH):
    raise RuntimeError(f"ONNX модель не найдена: {ONNX_PATH}")

ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

app = FastAPI(title="GNN Route Planner API", description="Returns route and estimated time using ONNX GNN policy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Глобальные временные константы (курьер)
# ----------------------------
COURIER_START_TIME = 540   # 09:00 в минутах
COURIER_END_TIME = 1260    # 21:00 в минутах

# ----------------------------
# OSRM: получение реального расстояния
# ----------------------------
def get_osrm_distance(lat1, lon1, lat2, lon2, profile="driving"):
    try:
        url = f"http://router.project-osrm.org/route/v1/{profile}/{lon1},{lat1};{lon2},{lat2}?overview=false"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("routes"):
                distance_m = data["routes"][0]["distance"]
                return distance_m / 1000.0  # в км
        return haversine((lat1, lon1), (lat2, lon2))
    except Exception as e:
        print(f"⚠️ OSRM error for ({lat1},{lon1})→({lat2},{lon2}): {e}")
        return haversine((lat1, lon1), (lat2, lon2))

class NodeInput(BaseModel):
    lat: float
    lon: float
    is_vip: int
    work_start: int
    work_end: int
    lunch_start: int
    lunch_end: int
    address: str  # <-- ДОБАВЛЕНО

class RouteRequest(BaseModel):
    nodes: List[NodeInput]

class RouteResponse(BaseModel):
    route: List[int]
    addresses: List[str]  # <-- ДОБАВЛЕНО
    total_time_minutes: float

def build_graph_from_nodes(nodes: List[NodeInput]):
    N = len(nodes)
    if N == 0:
        raise ValueError("Empty node list")
    
    coords = [(node.lat, node.lon) for node in nodes]
    is_vip = np.array([node.is_vip for node in nodes], dtype=np.int32)
    work_start = np.array([node.work_start for node in nodes], dtype=np.float32)
    work_end = np.array([node.work_end for node in nodes], dtype=np.float32)
    lunch_start = np.array([node.lunch_start for node in nodes], dtype=np.float32)
    lunch_end = np.array([node.lunch_end for node in nodes], dtype=np.float32)
    addresses = [node.address for node in nodes]  # <-- СОХРАНЯЕМ АДРЕСА

    x_np = np.array([
        [node.lat, node.lon, node.is_vip, node.work_start, node.work_end, node.lunch_start, node.lunch_end]
        for node in nodes
    ], dtype=np.float32)

    edge_index = []
    edge_attr = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_index.append([i, j])
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                haversine_dist = haversine((lat1, lon1), (lat2, lon2))
                osrm_dist = get_osrm_distance(lat1, lon1, lat2, lon2, profile="driving")
                edge_attr.append([haversine_dist, osrm_dist])

    edge_index = np.array(edge_index, dtype=np.int64).T  # [2, E]
    edge_attr = np.array(edge_attr, dtype=np.float32)    # [E, 2]

    return {
        "x": x_np,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "coords": coords,
        "work_start": work_start,
        "work_end": work_end,
        "lunch_start": lunch_start,
        "lunch_end": lunch_end,
        "is_vip": is_vip,
        "addresses": addresses,  # <-- ПЕРЕДАЁМ В ГРАФ
        "N": N
    }

def get_route_onnx_inference(graph_data, speed_kmh=30.0, service_time_min=30.0):
    x = graph_data["x"]
    edge_index = graph_data["edge_index"]
    edge_attr = graph_data["edge_attr"]
    coords = graph_data["coords"]
    work_start = graph_data["work_start"]
    work_end = graph_data["work_end"]
    lunch_start = graph_data["lunch_start"]
    lunch_end = graph_data["lunch_end"]
    N = graph_data["N"]

    visited = np.zeros(N, dtype=bool)
    current = 0
    visited[current] = True
    route = [current]
    current_time = float(COURIER_START_TIME)

    # Обслуживание в начальной точке (например, выезд из депо)
    current_time += service_time_min
    if current_time >= COURIER_END_TIME:
        total_time = current_time - COURIER_START_TIME
        return route, total_time

    for step in range(N - 1):
        ort_inputs = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr
        }
        logits = ort_session.run(None, ort_inputs)[0]

        mask = ~visited
        logits_masked = np.where(mask, logits, -1e9)
        next_node = int(np.argmax(logits_masked))

        # Расчёт времени до следующего узла
        lat1, lon1 = coords[current]
        lat2, lon2 = coords[next_node]
        d_km = get_osrm_distance(lat1, lon1, lat2, lon2, profile="driving")
        travel_time_min = d_km / speed_kmh * 60
        arrival_time = current_time + travel_time_min
        finish_time_at_next = arrival_time + service_time_min

        # 🔴 Прекращаем маршрут, если не успеваем обслужить до 21:00
        if finish_time_at_next > COURIER_END_TIME:
            break

        # Проверка временного окна КЛИЕНТА
        ws = work_start[next_node]
        we = work_end[next_node]
        ls = lunch_start[next_node]
        le = lunch_end[next_node]

        if arrival_time < ws:
            arrival_time = ws

        current_time = arrival_time + service_time_min
        visited[next_node] = True
        route.append(next_node)
        current = next_node

    total_time = current_time - COURIER_START_TIME
    return route, total_time

@app.post("/route", response_model=RouteResponse)
async def get_route(request: RouteRequest):
    try:
        graph_data = build_graph_from_nodes(request.nodes)
        route, total_time = get_route_onnx_inference(graph_data)
        
        # Получаем адреса в порядке маршрута
        addresses_in_route = [graph_data["addresses"][i] for i in route]
        
        return RouteResponse(
            route=route,
            addresses=addresses_in_route,
            total_time_minutes=round(total_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки запроса: {str(e)}")

def get_osrm_route_geometry(lat1, lon1, lat2, lon2, profile="driving"):
    """Возвращает геометрию маршрута между двумя точками как список [[lat, lon], ...]"""
    try:
        url = f"http://router.project-osrm.org/route/v1/{profile}/{lon1},{lat1};{lon2},{lat2}?geometries=geojson&overview=full"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("routes"):
                coords_geojson = data["routes"][0]["geometry"]["coordinates"]
                return [[lat, lon] for lon, lat in coords_geojson]
        return [[lat1, lon1], [lat2, lon2]]
    except Exception as e:
        print(f"⚠️ OSRM geometry error: {e}")
        return [[lat1, lon1], [lat2, lon2]]

@app.post("/route-geometry", response_model=List[List[List[float]]])
async def get_route_geometry(request: RouteRequest):
    """
    Принимает те же данные, что и /route,
    возвращает геометрию маршрута: список сегментов, каждый — список [lat, lon].
    """
    try:
        graph_data = build_graph_from_nodes(request.nodes)
        route, _ = get_route_onnx_inference(graph_data)

        coords = graph_data["coords"]
        geometry_segments = []
        for i in range(len(route) - 1):
            start_idx = route[i]
            end_idx = route[i + 1]
            lat1, lon1 = coords[start_idx]
            lat2, lon2 = coords[end_idx]
            segment = get_osrm_route_geometry(lat1, lon1, lat2, lon2)
            geometry_segments.append(segment)
        return geometry_segments
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка геометрии маршрута: {str(e)}")
    
def solve_tsp_with_constraints(graph_data, speed_kmh=30.0, service_time_min=30.0):
    """
    Жадный TSP с учётом временных окон и рабочего времени курьера.
    Возвращает маршрут (список индексов) и общее время.
    """
    coords = graph_data["coords"]
    work_start = graph_data["work_start"]
    work_end = graph_data["work_end"]
    lunch_start = graph_data["lunch_start"]
    lunch_end = graph_data["lunch_end"]
    N = graph_data["N"]

    if N == 0:
        return [], 0.0

    visited = [False] * N
    route = []
    current = 0  # начинаем с первого узла (депо)
    visited[current] = True
    route.append(current)
    current_time = float(COURIER_START_TIME)

    # Обслуживание начальной точки
    current_time += service_time_min
    if current_time > COURIER_END_TIME:
        return route, current_time - COURIER_START_TIME

    for _ in range(N - 1):
        best_next = -1
        best_arrival_time = float('inf')
        best_finish_time = float('inf')

        for j in range(N):
            if visited[j]:
                continue

            # Расчёт времени до j
            lat1, lon1 = coords[current]
            lat2, lon2 = coords[j]
            d_km = get_osrm_distance(lat1, lon1, lat2, lon2, profile="driving")
            travel_time = d_km / speed_kmh * 60
            arrival_time = current_time + travel_time
            finish_time = arrival_time + service_time_min

            # Пропускаем, если не успеваем закончить до 21:00
            if finish_time > COURIER_END_TIME:
                continue

            # Если приехали до начала рабочего дня — ждём
            if arrival_time < work_start[j]:
                arrival_time = work_start[j]
                finish_time = arrival_time + service_time_min
                if finish_time > COURIER_END_TIME:
                    continue

            # Проверка: не попадаем ли в обед?
            # (опционально: можно разрешить, но в реальности — нельзя)
            if lunch_start[j] <= arrival_time < lunch_end[j]:
                # Приехали в обед — пропускаем этого клиента сейчас
                continue

            # Выбираем ближайшего по времени прибытия (жадно)
            if arrival_time < best_arrival_time:
                best_arrival_time = arrival_time
                best_finish_time = finish_time
                best_next = j

        if best_next == -1:
            # Нет допустимого следующего клиента — завершаем маршрут
            break

        visited[best_next] = True
        route.append(best_next)
        current = best_next
        current_time = best_finish_time

    total_time = current_time - COURIER_START_TIME
    return route, total_time

@app.post("/comy_voyager", response_model=RouteResponse)
async def get_tsp_route(request: RouteRequest):
    """
    Эндпоинт для построения маршрута методом коммивояжёра (жадный TSP с ограничениями).
    """
    try:
        graph_data = build_graph_from_nodes(request.nodes)
        route, total_time = solve_tsp_with_constraints(graph_data)

        addresses_in_route = [graph_data["addresses"][i] for i in route]

        return RouteResponse(
            route=route,
            addresses=addresses_in_route,
            total_time_minutes=round(total_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка TSP-маршрута: {str(e)}")