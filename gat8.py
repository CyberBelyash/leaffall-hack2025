import torch
import torch.onnx
import onnxruntime as ort
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch.distributions import Categorical
import pandas as pd
from haversine import haversine
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from itertools import permutations
import requests
from functools import lru_cache
import folium
from folium import Marker
from scipy.spatial import cKDTree

# ----------------------------
# Путь к файлу
# ----------------------------
file_path = r"C:\Users\eliza\Downloads\test_list.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Файл не найден: {file_path}")

# ----------------------------
# OSRM: получение реального расстояния
# ----------------------------
def get_osrm_distance(lat1, lon1, lat2, lon2, profile="driving"):
    try:
        url = f"http://router.project-osrm.org/route/v1/{profile}/{lon1},{lat1};{lon2},{lat2}?overview=false&annotations=false"
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

# ----------------------------
# 1. Загрузка и предобработка
# ----------------------------
df = pd.read_excel(file_path)[:20]

def time_to_minutes(t):
    h, m = map(int, t.split(':'))
    return h * 60 + m

df['work_start'] = df['Время начала рабочего дня'].apply(time_to_minutes)
df['work_end']   = df['Время окончания рабочего дня'].apply(time_to_minutes)
df['lunch_start'] = df['Время начала обеда'].apply(time_to_minutes)
df['lunch_end']   = df['Время окончания обеда'].apply(time_to_minutes)
df['is_vip'] = (df['Уровень клиента'] == 'VIP').astype(int)

work_start = df['work_start'].values
work_end   = df['work_end'].values
lunch_start = df['lunch_start'].values
lunch_end   = df['lunch_end'].values
is_vip = df['is_vip'].values

node_features = df[[
    'Географическая широта', 'Географическая долгота',
    'is_vip', 'work_start', 'work_end', 'lunch_start', 'lunch_end'
]].values
x = torch.tensor(node_features, dtype=torch.float)
N = len(df)
coords = list(zip(df['Географическая широта'], df['Географическая долгота']))

# ----------------------------
# 2. Создание ПОЛНОСВЯЗНОГО графа с двумя признаками рёбер: [haversine, osrm]
# ----------------------------
print("🔍 Получение реальных расстояний через OSRM (может занять несколько минут)...")
edge_index = []
edge_attr = []

for i in range(N):
    for j in range(N):
        if i != j:
            edge_index.append([i, j])
            haversine_dist = haversine(coords[i], coords[j])
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            osrm_dist = get_osrm_distance(lat1, lon1, lat2, lon2, profile="driving")
            edge_attr.append([haversine_dist, osrm_dist])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

# Матрица OSRM-расстояний для быстрого доступа
osrm_dist_matrix = np.full((N, N), np.inf, dtype=np.float32)
for (i, j), attr in zip(edge_index.t().numpy(), edge_attr.numpy()):
    osrm_dist_matrix[i, j] = attr[1]

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
print(f"✅ Граф построен: {N} узлов, {edge_index.size(1)} рёбер.")

# ----------------------------
# 3. GNN Policy Network (edge_attr_dim=2)
# ----------------------------
class GNNPolicy(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_attr_dim=2):
        super().__init__()
        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=2, concat=True, edge_dim=edge_attr_dim)
        self.conv2 = TransformerConv(hidden_dim * 2, out_dim, heads=1, concat=False, edge_dim=edge_attr_dim)
        self.decoder = nn.Linear(out_dim, 1)

    def encode(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = self.conv2(h, edge_index, edge_attr)
        return h

    def forward(self, x, edge_index, edge_attr, mask=None):
        h = self.encode(x, edge_index, edge_attr)
        logits = self.decoder(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits

# ----------------------------
# 4. REINFORCE с OSRM-расстоянием и временем обслуживания
# ----------------------------
def reinforce_batch_step(model, data, coords, osrm_dist_matrix, work_start, work_end, lunch_start, lunch_end, is_vip,
                        speed_kmh=30.0, num_samples=8, entropy_coef=0.01, time_window_penalty=1e6, service_time_min=30.0):
    N = data.x.size(0)
    node_embeddings = model.encode(data.x, data.edge_index, data.edge_attr)

    total_times = []
    total_rewards = []
    log_prob_sums = []
    entropy_sums = []

    for _ in range(num_samples):
        visited = torch.zeros(N, dtype=torch.bool)
        current = random.randint(0, N - 1)
        visited[current] = True
        log_probs = []
        entropies = []

        current_time = 540.0  # 09:00
        start_time = current_time
        violated = False
        vip_count = 0

        if is_vip[current]:
            vip_count += 1

        # Обслуживание в стартовой точке
        current_time += service_time_min

        for step in range(N - 1):
            logits = model.decoder(node_embeddings).squeeze(-1)
            mask = ~visited
            logits = logits.masked_fill(~mask, -1e9)
            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
            next_node = dist.sample()

            log_probs.append(dist.log_prob(next_node))
            entropies.append(dist.entropy())

            d_km = osrm_dist_matrix[current, next_node.item()]
            travel_time_min = d_km / speed_kmh * 60
            arrival_time = current_time + travel_time_min

            ws = work_start[next_node.item()]
            we = work_end[next_node.item()]
            ls = lunch_start[next_node.item()]
            le = lunch_end[next_node.item()]

            if arrival_time < ws or arrival_time > we or (ls <= arrival_time <= le):
                violated = True
            elif arrival_time < ws:
                arrival_time = ws

            current_time = arrival_time + service_time_min
            visited[next_node] = True
            current = next_node.item()

            if is_vip[next_node.item()]:
                vip_count += 1

        total_time_elapsed = current_time - start_time
        total_times.append(total_time_elapsed)

        base_reward = -current_time
        reward = base_reward
        if vip_count > 0:
            reward += 0.1 * vip_count * abs(base_reward)
        if violated:
            reward -= time_window_penalty

        total_rewards.append(reward)
        log_prob_sums.append(torch.stack(log_probs).sum())
        entropy_sums.append(torch.stack(entropies).sum())

    rewards = torch.tensor(total_rewards, dtype=torch.float)
    log_prob_sums = torch.stack(log_prob_sums)
    entropy_sums = torch.stack(entropy_sums)

    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    policy_loss = -(advantage * log_prob_sums).mean()
    entropy_loss = -entropy_sums.mean()
    loss = policy_loss + entropy_coef * entropy_loss

    avg_time_min = np.mean(total_times)
    return loss, avg_time_min

# ----------------------------
# 5. Baseline (жадный по haversine)
# ----------------------------
def compute_greedy_tsp_time(coords, speed_kmh=30.0, service_time_min=30.0):
    N = len(coords)
    min_time = float('inf')
    for start in range(N):
        visited = [False] * N
        current = start
        visited[current] = True
        total_dist = 0.0
        # Время: старт + обслуживание первого
        current_time = 540.0 + service_time_min
        for _ in range(N - 1):
            nearest = -1
            nearest_dist = float('inf')
            for j in range(N):
                if not visited[j]:
                    d = haversine(coords[current], coords[j])
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest = j
            total_dist += nearest_dist
            travel_time = nearest_dist / speed_kmh * 60
            current_time += travel_time + service_time_min
            visited[nearest] = True
            current = nearest
        total_time = current_time - 540.0
        if total_time < min_time:
            min_time = total_time
    return min_time

optimal_time = compute_greedy_tsp_time(coords, speed_kmh=30.0, service_time_min=30.0)
print(f"🟢 Жадное время (haversine + 30 мин/узел): {optimal_time:.1f} мин")

# ----------------------------
# 6. Обучение
# ----------------------------
device = torch.device('cpu')
model = GNNPolicy(in_dim=7, hidden_dim=32, out_dim=32, edge_attr_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
avg_times = []

print("🚀 Начало обучения с учётом временных окон, OSRM и 30 мин обслуживания...")
for epoch in range(1000):
    loss, avg_time = reinforce_batch_step(
        model, data, coords, osrm_dist_matrix, work_start, work_end,
        lunch_start, lunch_end, is_vip,
        speed_kmh=30.0, num_samples=2, time_window_penalty=10000,
        service_time_min=30.0
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    losses.append(loss.item())
    avg_times.append(avg_time)
    
    if epoch % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():8.2f} | Avg Time: {avg_time:6.1f} min")

print("\n✅ Обучение завершено.")

# ----------------------------
# 7. Маршрутизация (PyTorch)
# ----------------------------
def get_route(model, data, coords, osrm_dist_matrix, work_start, work_end, lunch_start, lunch_end, is_vip,
              speed_kmh=30.0, service_time_min=30.0, start_time_minutes=540, end_time_minutes=1260):
    N = data.x.size(0)
    visited = torch.zeros(N, dtype=torch.bool)
    current = 0
    visited[current] = True
    route = [current]
    current_time = float(start_time_minutes)
    violated = False
    vip_count = int(is_vip[current])

    # Обслуживание в стартовой точке
    current_time += service_time_min
    if current_time >= end_time_minutes:
        print("⚠️ Превышено конечное время уже на первом узле.")
        return route, current_time - start_time_minutes

    for step in range(N - 1):
        h = model.encode(data.x, data.edge_index, data.edge_attr)
        mask = ~visited
        logits = model.decoder(h).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        next_node = logits.argmax().item()

        d_km = osrm_dist_matrix[current, next_node]
        travel_time_min = d_km / speed_kmh * 60
        arrival_time = current_time + travel_time_min
        finish_time = arrival_time + service_time_min

        # Проверка: если даже прибытие + обслуживание выходит за end_time — остановка
        if finish_time > end_time_minutes:
            print(f"🛑 Прекращение маршрута на шаге {step+1}: finish_time={finish_time:.1f} > end_time={end_time_minutes}")
            break

        # Проверка временных окон
        ws = work_start[next_node]
        we = work_end[next_node]
        ls = lunch_start[next_node]
        le = lunch_end[next_node]

        if arrival_time < ws or arrival_time > we or (ls <= arrival_time <= le):
            violated = True
        elif arrival_time < ws:
            arrival_time = ws

        current_time = arrival_time + service_time_min
        visited[next_node] = True
        route.append(next_node)
        current = next_node

        if is_vip[next_node]:
            vip_count += 1

    status = "❌ Нарушение окон" if violated else "✅ В рамках окон"
    print(f"   {status} | VIP посещено: {vip_count} | Посещено узлов: {len(route)}")
    return route, current_time - start_time_minutes

# ----------------------------
# 8. Маршрутизация (ONNX)
# ----------------------------
def get_route_onnx(ort_session, data, coords, osrm_dist_matrix, work_start, work_end, lunch_start, lunch_end, is_vip,
                   speed_kmh=30.0, service_time_min=30.0, start_time_minutes=540, end_time_minutes=1260):
    N = data.x.size(0)
    visited = np.zeros(N, dtype=bool)
    current = 0
    visited[current] = True
    route = [current]
    current_time = float(start_time_minutes)
    violated = False
    vip_count = int(is_vip[current])

    current_time += service_time_min
    if current_time >= end_time_minutes:
        print("⚠️ Превышено конечное время уже на первом узле.")
        return route, current_time - start_time_minutes

    x_np = data.x.cpu().numpy().astype(np.float32)
    edge_index_np = data.edge_index.cpu().numpy().astype(np.int64)
    edge_attr_np = data.edge_attr.cpu().numpy().astype(np.float32)

    for step in range(N - 1):
        ort_inputs = {
            'x': x_np,
            'edge_index': edge_index_np,
            'edge_attr': edge_attr_np
        }
        logits = ort_session.run(None, ort_inputs)[0]
        mask = ~visited
        logits_masked = np.where(mask, logits, -1e9)
        next_node = int(np.argmax(logits_masked))

        d_km = osrm_dist_matrix[current, next_node]
        travel_time_min = d_km / speed_kmh * 60
        arrival_time = current_time + travel_time_min
        finish_time = arrival_time + service_time_min

        if finish_time > end_time_minutes:
            print(f"🛑 Прекращение маршрута на шаге {step+1}: finish_time={finish_time:.1f} > end_time={end_time_minutes}")
            break

        ws = work_start[next_node]
        we = work_end[next_node]
        ls = lunch_start[next_node]
        le = lunch_end[next_node]

        if arrival_time < ws or arrival_time > we or (ls <= arrival_time <= le):
            violated = True
        elif arrival_time < ws:
            arrival_time = ws

        current_time = arrival_time + service_time_min
        visited[next_node] = True
        route.append(next_node)
        current = next_node

        if is_vip[next_node]:
            vip_count += 1

    status = "❌ Нарушение окон" if violated else "✅ В рамках окон"
    print(f"   {status} | VIP посещено: {vip_count} | Посещено узлов: {len(route)}")
    return route, current_time - start_time_minutes

# ----------------------------
# 9. Запуск маршрутизации
# ----------------------------
final_route, final_time = get_route(
    model, data, coords, osrm_dist_matrix, work_start, work_end,
    lunch_start, lunch_end, is_vip, speed_kmh=30.0, service_time_min=30.0
)
print(f"\n🏁 Финальный маршрут (PyTorch):")
print(" -> ".join(str(i + 1) for i in final_route))
print(f"⏱️ Итоговое время: {final_time:.1f} минут")

# ----------------------------
# 10. Экспорт в ONNX
# ----------------------------
def export_model_to_onnx(model, data, filepath, opset_version=15):
    model.eval()
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x, edge_index, edge_attr):
            return self.model(x, edge_index, edge_attr, mask=None)
    wrapped_model = ONNXWrapper(model)
    torch.onnx.export(
        wrapped_model,
        (data.x, data.edge_index, data.edge_attr),
        filepath,
        input_names=['x', 'edge_index', 'edge_attr'],
        output_names=['logits'],
        dynamic_axes={
            'x': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'},
            'edge_attr': {0: 'num_edges'},
            'logits': {0: 'num_nodes'}
        },
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True
    )
    print(f"✅ Модель экспортирована в ONNX: {filepath}")

export_model_to_onnx(model, data, "gat3_policy_model.onnx")
ort_session = ort.InferenceSession("gat3_policy_model.onnx", providers=['CPUExecutionProvider'])

onnx_route, onnx_time = get_route_onnx(
    ort_session, data, coords, osrm_dist_matrix, work_start, work_end,
    lunch_start, lunch_end, is_vip, speed_kmh=30.0, service_time_min=30.0
)
print(f"\n🏁 Маршрут (ONNX):")
print(" -> ".join(str(i + 1) for i in onnx_route))
print(f"⏱️ Время (ONNX): {onnx_time:.1f} минут")

print(f"\nСравнение:")
print(f"PyTorch: {final_time:.1f} мин")
print(f"ONNX   : {onnx_time:.1f} мин")
print("✅ Результаты совпадают!" if abs(final_time - onnx_time) < 1e-3 else "⚠️ Расхождение!")

# ----------------------------
# 11. Визуализация с OSRM-геометрией
# ----------------------------
def get_osrm_route_segment(coords, start_idx, end_idx, profile="driving"):
    start_lat, start_lon = coords[start_idx]
    end_lat, end_lon = coords[end_idx]
    url = f"http://router.project-osrm.org/route/v1/{profile}/{start_lon},{start_lat};{end_lon},{end_lat}?geometries=geojson&overview=full&annotations=true"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"OSRM segment request failed: {resp.status_code}")
    data = resp.json()
    route = data["routes"][0]
    geometry = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
    distance_m = route["distance"]
    distance_km = distance_m / 1000.0
    return geometry, distance_km

def visualize_route_on_map(coords, path, profile="driving", output_file="route_map.html"):
    if len(path) < 2:
        raise ValueError("Маршрут должен содержать хотя бы две точки.")
    start_lat, start_lon = coords[path[0]]
    m = folium.Map(location=[start_lat, start_lon], zoom_start=13)
    for idx, (lat, lon) in enumerate(coords):
        color = "red" if idx in path else "blue"
        Marker(location=[lat, lon], popup=f"Point {idx+1}", icon=folium.Icon(color=color)).add_to(m)
    for order, idx in enumerate(path):
        Marker(
            location=coords[idx],
            icon=folium.DivIcon(html=f"""<div style="color: white; font-weight: bold; 
                                          background: red; border-radius: 50%; 
                                          width: 24px; height: 24px; 
                                          display: flex; align-items: center; 
                                          justify-content: center;">{order}</div>""")
        ).add_to(m)
    for i in range(len(path) - 1):
        start_idx = path[i]
        end_idx = path[i + 1]
        try:
            geometry, dist_km = get_osrm_route_segment(coords, start_idx, end_idx, profile)
            tooltip_text = f"Сегмент {i+1} → {i+2}<br>Расстояние: {dist_km:.3f} км"
            folium.PolyLine(locations=geometry, color="green", weight=5, opacity=0.8, tooltip=tooltip_text).add_to(m)
        except Exception as e:
            print(f"⚠️ Ошибка сегмента {start_idx}→{end_idx}: {e}")
            folium.PolyLine(
                locations=[coords[start_idx], coords[end_idx]],
                color="orange",
                weight=3,
                dash_array='5,5',
                tooltip="OSRM недоступен"
            ).add_to(m)
    m.save(output_file)
    print(f"✅ Карта сохранена: {output_file}")

visualize_route_on_map(coords, onnx_route, profile="driving", output_file="final_route_osrm.html")

# ----------------------------
# 12. Графики обучения
# ----------------------------
epochs = list(range(1000))
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, color='tab:red')
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, avg_times, color='tab:blue', label='GNN Policy (avg)')
plt.axhline(y=optimal_time, color='green', linestyle='--', linewidth=2, label=f'Greedy (haversine + 30 мин): {optimal_time:.1f} min')
plt.axhline(y=final_time, color='orange', linestyle='-.', linewidth=2, label=f'Final Route: {final_time:.1f} min')
plt.title('Route Time (min) vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Time (minutes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()