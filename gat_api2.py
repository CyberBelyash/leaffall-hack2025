import numpy as np
from haversine import haversine
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional
import onnxruntime as ort
import os
import requests
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# Конфигурация БД и JWT
# ----------------------------
DB_CONFIG = {
    "database": "postgres",
    "user": "postgres",
    "password": "123456789",
    "host": "localhost",
    "port": "5432"
}

SECRET_KEY = "gnnplanner-secret-2025-dyadchenko"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 часа

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

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
        logger.warning(f"⚠️ OSRM error for ({lat1},{lon1})→({lat2},{lon2}): {e}")
        return haversine((lat1, lon1), (lat2, lon2))

# ----------------------------
# Pydantic models
# ----------------------------
class NodeInput(BaseModel):
    lat: float
    lon: float
    is_vip: int
    work_start: int
    work_end: int
    lunch_start: int
    lunch_end: int
    address: str

class RouteRequest(BaseModel):
    nodes: List[NodeInput]

class RouteResponse(BaseModel):
    route: List[int]
    addresses: List[str]
    total_time_minutes: float

class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    email: str
    password: str
    name: str

class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"

# ----------------------------
# Auth helpers
# ----------------------------
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def init_routes_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_routes (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            route_name VARCHAR(255),
            nodes JSONB NOT NULL,
            route_order INTEGER[] NOT NULL,
            addresses TEXT[] NOT NULL,
            total_time_minutes FLOAT NOT NULL,
            algorithm_used VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Неверный или отсутствующий токен",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if email is None or user_id is None:
            raise credentials_exception
        return {"email": email, "user_id": user_id}
    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise credentials_exception

# ----------------------------
# Startup
# ----------------------------
@app.on_event("startup")
async def startup_event():
    init_db()
    init_routes_table()

# ----------------------------
# Auth endpoints
# ----------------------------
@app.post("/register", response_model=TokenResponse)
async def register(user: UserRegister):
    if not user.email or not user.password or not user.name:
        raise HTTPException(status_code=400, detail="Все поля обязательны")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE email = %s", (user.email,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Пользователь с таким email уже существует")
        hashed_pw = get_password_hash(user.password)
        cur.execute(
            "INSERT INTO users (email, name, hashed_password) VALUES (%s, %s, %s) RETURNING id",
            (user.email, user.name, hashed_pw)
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        token = create_access_token({"sub": user.email, "user_id": user_id})
        return TokenResponse(token=token)
    except Exception as e:
        conn.rollback()
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка регистрации: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, email, name, hashed_password FROM users WHERE email = %s", (user.email,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Неверный email или пароль")
        user_id, email, name, hashed_pw = row
        if not verify_password(user.password, hashed_pw):
            raise HTTPException(status_code=401, detail="Неверный email или пароль")
        token = create_access_token({"sub": email, "user_id": user_id})
        return TokenResponse(token=token)
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка входа: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.get("/me")
async def me(user = Depends(get_current_user)):
    return {"user_id": user["user_id"], "email": user["email"], "name": user["email"].split('@')[0]}


# ----------------------------
# Admin: проверка прав
# ----------------------------
def get_admin_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=403,
        detail="Доступ запрещён",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if email is None or user_id is None:
            raise credentials_exception
        # Проверка is_admin
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT is_admin FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row or not row[0]:
            raise credentials_exception
        return {"email": email, "user_id": user_id}
    except JWTError as e:
        logger.error(f"Admin JWT error: {e}")
        raise credentials_exception

# ----------------------------
# Admin endpoints
# ----------------------------
@app.get("/admin/users")
async def admin_get_users(admin = Depends(get_admin_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, email, name, created_at, is_admin FROM users ORDER BY id
        """)
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "email": r[1],
                "name": r[2],
                "created_at": r[3].isoformat(),
                "is_admin": r[4]
            }
            for r in rows
        ]
    finally:
        cur.close()
        conn.close()

@app.post("/admin/users")
async def admin_create_user(
    data: dict,
    admin = Depends(get_admin_user)
):
    email = data.get("email")
    name = data.get("name")
    password = data.get("password")
    is_admin = bool(data.get("is_admin", False))
    if not email or not name or not password:
        raise HTTPException(400, "email, name, password обязательны")

    hashed_pw = get_password_hash(password)
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO users (email, name, hashed_password, is_admin)
            VALUES (%s, %s, %s, %s) RETURNING id
        """, (email, name, hashed_pw, is_admin))
        user_id = cur.fetchone()[0]
        conn.commit()
        return {"id": user_id, "email": email, "name": name, "is_admin": is_admin}
    except psycopg2.IntegrityError as e:
        conn.rollback()
        raise HTTPException(400, f"Ошибка: дубликат email или другая ошибка БД: {e}")
    finally:
        cur.close()
        conn.close()

@app.put("/admin/users/{user_id}")
async def admin_update_user(
    user_id: int, data: dict, admin = Depends(get_admin_user)
):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Проверяем наличие пользователя
        cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
        if not cur.fetchone():
            raise HTTPException(404, "Пользователь не найден")

        update_fields = []
        params = []

        # Обновляем email, только если он передан и не пустой
        email = data.get("email")
        if email is not None and email.strip() != "":
            update_fields.append("email = %s")
            params.append(email.strip())

        # Обновляем name, только если он передан и не пустой
        name = data.get("name")
        if name is not None and name.strip() != "":
            update_fields.append("name = %s")
            params.append(name.strip())

        # Обновляем пароль, только если он передан и не пустой
        password = data.get("password")
        if password is not None and password.strip() != "":
            update_fields.append("hashed_password = %s")
            params.append(get_password_hash(password.strip()))

        # is_admin обновляем даже если передан как false (но не если отсутствует)
        if "is_admin" in data:
            is_admin = bool(data["is_admin"])
            update_fields.append("is_admin = %s")
            params.append(is_admin)

        # Если ни одно поле не подошло для обновления — выходим
        if not update_fields:
            raise HTTPException(400, "Нет валидных полей для обновления")

        # Добавляем user_id в конец параметров
        params.append(user_id)

        # Формируем и выполняем запрос
        query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = %s"
        cur.execute(query, params)
        conn.commit()

        return {"status": "ok", "user_id": user_id}
    except Exception as e:
        logger.error(f"Update user error: {e}")
        raise HTTPException(500, f"Ошибка обновления пользователя: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: int, admin = Depends(get_admin_user)):
    if user_id == admin["user_id"]:
        raise HTTPException(400, "Нельзя удалить самого себя")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "Пользователь не найден")
        conn.commit()
        return {"status": "ok"}
    finally:
        cur.close()
        conn.close()

# ----------------------------
# Routes (saved_routes)
# ----------------------------
@app.get("/admin/routes")
async def admin_get_routes(admin = Depends(get_admin_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, user_id, route_name, algorithm_used, total_time_minutes, created_at
            FROM saved_routes
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "user_id": r[1],
                "route_name": r[2] or "—",
                "algorithm_used": r[3],
                "total_time_minutes": r[4],
                "created_at": r[5].isoformat()
            }
            for r in rows
        ]
    finally:
        cur.close()
        conn.close()

@app.put("/admin/routes/{route_id}")
async def admin_update_route(route_id: int, data: dict, admin = Depends(get_admin_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Сначала получаем user_id маршрута
        cur.execute("SELECT user_id FROM saved_routes WHERE id = %s", (route_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Маршрут не найден")
        route_owner_id = row[0]

        # 🔒 Запрещаем редактирование чужих маршрутов
        if route_owner_id != admin["user_id"]:
            raise HTTPException(
                status_code=403,
                detail="Редактирование чужих маршрутов запрещено. Админ может только просматривать и удалять."
            )

        # Дальнейшая логика обновления — без изменений
        updates = []
        params = []
        if "user_id" in data:
            updates.append("user_id = %s")
            params.append(data["user_id"])
        if "route_name" in data:
            updates.append("route_name = %s")
            params.append(data["route_name"])
        if "nodes" in data:
            updates.append("nodes = %s::jsonb")
            params.append(json.dumps(data["nodes"]))
        if "route_order" in data:
            updates.append("route_order = %s")
            params.append(data["route_order"])
        if "addresses" in data:
            updates.append("addresses = %s")
            params.append(data["addresses"])
        if "total_time_minutes" in data:
            updates.append("total_time_minutes = %s")
            params.append(data["total_time_minutes"])
        if "algorithm_used" in data:
            updates.append("algorithm_used = %s")
            params.append(data["algorithm_used"])
        if not updates:
            raise HTTPException(400, "Нет полей для обновления")

        params.append(route_id)
        query = f"UPDATE saved_routes SET {', '.join(updates)} WHERE id = %s"
        cur.execute(query, params)
        if cur.rowcount == 0:
            raise HTTPException(404, "Маршрут не найден")
        conn.commit()
        return {"status": "ok"}
    finally:
        cur.close()
        conn.close()

@app.post("/admin/routes")
async def admin_create_route(data: dict, admin = Depends(get_admin_user)):
    required = ["user_id", "nodes", "route_order", "addresses", "total_time_minutes", "algorithm_used"]
    for k in required:
        if k not in data:
            raise HTTPException(400, f"Поле '{k}' обязательно")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO saved_routes (
                user_id, route_name, nodes, route_order, addresses, total_time_minutes, algorithm_used
            ) VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s) RETURNING id
        """, (
            data["user_id"],
            data.get("route_name"),
            json.dumps(data["nodes"]),
            data["route_order"],
            data["addresses"],
            data["total_time_minutes"],
            data["algorithm_used"]
        ))
        route_id = cur.fetchone()[0]
        conn.commit()
        return {"id": route_id}
    finally:
        cur.close()
        conn.close()

@app.put("/admin/routes/{route_id}")
async def admin_update_route(route_id: int, data: dict, admin = Depends(get_admin_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        updates = []
        params = []
        if "user_id" in data:
            updates.append("user_id = %s")
            params.append(data["user_id"])
        if "route_name" in data:
            updates.append("route_name = %s")
            params.append(data["route_name"])
        if "nodes" in data:
            updates.append("nodes = %s::jsonb")
            params.append(json.dumps(data["nodes"]))
        if "route_order" in data:
            updates.append("route_order = %s")
            params.append(data["route_order"])
        if "addresses" in data:
            updates.append("addresses = %s")
            params.append(data["addresses"])
        if "total_time_minutes" in data:
            updates.append("total_time_minutes = %s")
            params.append(data["total_time_minutes"])
        if "algorithm_used" in data:
            updates.append("algorithm_used = %s")
            params.append(data["algorithm_used"])
        if not updates:
            raise HTTPException(400, "Нет полей для обновления")
        params.append(route_id)
        query = f"UPDATE saved_routes SET {', '.join(updates)} WHERE id = %s"
        cur.execute(query, params)
        if cur.rowcount == 0:
            raise HTTPException(404, "Маршрут не найден")
        conn.commit()
        return {"status": "ok"}
    finally:
        cur.close()
        conn.close()

@app.delete("/admin/routes/{route_id}")
async def admin_delete_route(route_id: int, admin = Depends(get_admin_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM saved_routes WHERE id = %s", (route_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "Маршрут не найден")
        conn.commit()
        return {"status": "ok"}
    finally:
        cur.close()
        conn.close()
        
        
# ----------------------------
# Route saving endpoints
# ----------------------------
@app.post("/save-route")
async def save_route(route_data: dict, user = Depends(get_current_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        nodes_data = route_data["nodes"]
        route_order_data = route_data["route_order"]
        addresses_data = route_data["addresses"]

        # Универсальная конвертация в списки
        if not isinstance(route_order_data, list):
            route_order_data = list(route_order_data)
        if not isinstance(addresses_data, list):
            addresses_data = list(addresses_data)

        cur.execute("""
            INSERT INTO saved_routes (
                user_id, route_name, nodes, route_order, addresses, total_time_minutes, algorithm_used
            ) VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s)
            RETURNING id, created_at
        """, (
            user["user_id"],
            route_data.get("route_name"),
            json.dumps(nodes_data),
            route_order_data,
            addresses_data,
            route_data["total_time_minutes"],
            route_data["algorithm_used"]
        ))
        route_id, created_at = cur.fetchone()
        conn.commit()
        return {
            "id": route_id,
            "created_at": created_at.isoformat(),
            "message": "Маршрут успешно сохранён"
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Save route error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения маршрута: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.get("/my-routes")
async def get_user_routes(user = Depends(get_current_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, route_name, algorithm_used, total_time_minutes, created_at
            FROM saved_routes
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user["user_id"],))
        routes = cur.fetchall()
        return [{
            "id": r[0],
            "route_name": r[1] or "Без названия",
            "algorithm_used": r[2],
            "total_time_minutes": r[3],
            "created_at": r[4].isoformat()
        } for r in routes]
    except Exception as e:
        logger.error(f"My routes fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения маршрутов: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.get("/route/{route_id}")
async def get_route_by_id(route_id: int, user = Depends(get_current_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT nodes, route_order, addresses, total_time_minutes, algorithm_used
            FROM saved_routes
            WHERE id = %s AND user_id = %s
        """, (route_id, user["user_id"]))
        result = cur.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Маршрут не найден или нет доступа")
        
        nodes_json, route_order, addresses, total_time_minutes, algorithm_used = result

        # Универсальная обработка
        nodes = json.loads(nodes_json) if isinstance(nodes_json, str) else nodes_json
        route_order = list(route_order) if not isinstance(route_order, list) else route_order
        addresses = list(addresses) if not isinstance(addresses, list) else addresses

        return {
            "nodes": nodes,
            "route_order": route_order,
            "addresses": addresses,
            "total_time_minutes": total_time_minutes,
            "algorithm_used": algorithm_used
        }
    except Exception as e:
        logger.error(f"Route #{route_id} fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения маршрута: {str(e)}")
    finally:
        cur.close()
        conn.close()

# ----------------------------
# Graph construction
# ----------------------------
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
    addresses = [node.address for node in nodes]

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
                osrm_dist = get_osrm_distance(lat1, lon1, lat2, lon2)
                edge_attr.append([haversine_dist, osrm_dist])

    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_attr = np.array(edge_attr, dtype=np.float32)

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
        "addresses": addresses,
        "N": N
    }

# ----------------------------
# Route inference (GNN-based)
# ----------------------------
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
    current_time += service_time_min

    if current_time >= COURIER_END_TIME:
        return route, current_time - COURIER_START_TIME

    for _ in range(N - 1):
        ort_inputs = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr
        }
        logits = ort_session.run(None, ort_inputs)[0]
        mask = ~visited
        logits_masked = np.where(mask, logits, -1e9)
        next_node = int(np.argmax(logits_masked))

        lat1, lon1 = coords[current]
        lat2, lon2 = coords[next_node]
        d_km = get_osrm_distance(lat1, lon1, lat2, lon2)
        travel_time_min = d_km / speed_kmh * 60
        arrival_time = current_time + travel_time_min
        finish_time_at_next = arrival_time + service_time_min

        if finish_time_at_next > COURIER_END_TIME:
            break

        ws = graph_data["work_start"][next_node]
        we = graph_data["work_end"][next_node]
        ls = graph_data["lunch_start"][next_node]
        le = graph_data["lunch_end"][next_node]

        if arrival_time < ws:
            arrival_time = ws
        elif ls <= arrival_time < le:
            # Пропускаем узел, если он в обед
            continue

        current_time = arrival_time + service_time_min
        visited[next_node] = True
        route.append(next_node)
        current = next_node

    total_time = current_time - COURIER_START_TIME
    return route, total_time

# ----------------------------
# Route inference (Greedy TSP)
# ----------------------------
def solve_tsp_with_constraints(graph_data, speed_kmh=30.0, service_time_min=30.0):
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
    current = 0
    visited[current] = True
    route.append(current)

    current_time = float(COURIER_START_TIME)
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

            lat1, lon1 = coords[current]
            lat2, lon2 = coords[j]
            d_km = get_osrm_distance(lat1, lon1, lat2, lon2)
            travel_time = d_km / speed_kmh * 60
            arrival_time = current_time + travel_time
            finish_time = arrival_time + service_time_min

            if finish_time > COURIER_END_TIME:
                continue

            ws = work_start[j]
            we = work_end[j]
            ls = lunch_start[j]
            le = lunch_end[j]

            if arrival_time < ws:
                arrival_time = ws
                finish_time = arrival_time + service_time_min
                if finish_time > COURIER_END_TIME:
                    continue
            elif ls <= arrival_time < le:
                continue

            if arrival_time < best_arrival_time:
                best_arrival_time = arrival_time
                best_finish_time = finish_time
                best_next = j

        if best_next == -1:
            break

        visited[best_next] = True
        route.append(best_next)
        current = best_next
        current_time = best_finish_time

    total_time = current_time - COURIER_START_TIME
    return route, total_time

# ----------------------------
# OSRM geometry helper
# ----------------------------
def get_osrm_route_geometry(lat1, lon1, lat2, lon2, profile="driving"):
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
        logger.warning(f"OSRM geometry error: {e}")
        return [[lat1, lon1], [lat2, lon2]]

# ----------------------------
# Protected endpoints
# ----------------------------
@app.post("/route", response_model=RouteResponse)
async def get_route(request: RouteRequest, user = Depends(get_current_user)):
    try:
        graph_data = build_graph_from_nodes(request.nodes)
        route, total_time = get_route_onnx_inference(graph_data)
        addresses_in_route = [graph_data["addresses"][i] for i in route]
        return RouteResponse(
            route=route,
            addresses=addresses_in_route,
            total_time_minutes=round(total_time, 2)
        )
    except Exception as e:
        logger.error(f"GNN route error: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка GNN-маршрута: {str(e)}")

@app.post("/comy_voyager", response_model=RouteResponse)
async def get_tsp_route(request: RouteRequest, user = Depends(get_current_user)):
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
        logger.error(f"TSP route error: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка TSP-маршрута: {str(e)}")

@app.post("/route-geometry", response_model=List[List[List[float]]])
async def get_route_geometry(request: RouteRequest, user = Depends(get_current_user)):
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
        logger.error(f"Route geometry error: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка геометрии маршрута: {str(e)}")