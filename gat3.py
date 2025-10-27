import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from haversine import haversine
import pandas as pd
import numpy as np
import os
import random
from heapq import heappush, heappop

# ----------------------------
# Путь к файлу
# ----------------------------
file_path = r"C:\Users\eliza\Downloads\test_list.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Файл не найден: {file_path}")

# ----------------------------
# 1. Загрузка и предобработка
# ----------------------------
df = pd.read_excel(file_path)

def time_to_minutes(t):
    h, m = map(int, t.split(':'))
    return h * 60 + m

df['work_start'] = df['Время начала рабочего дня'].apply(time_to_minutes)
df['work_end']   = df['Время окончания рабочего дня'].apply(time_to_minutes)
df['lunch_start'] = df['Время начала обеда'].apply(time_to_minutes)
df['lunch_end']   = df['Время окончания обеда'].apply(time_to_minutes)
df['is_vip'] = (df['Уровень клиента'] == 'VIP').astype(int)

node_features = df[[
    'Географическая широта', 'Географическая долгота',
    'is_vip', 'work_start', 'work_end', 'lunch_start', 'lunch_end'
]].values
x = torch.tensor(node_features, dtype=torch.float)
N = len(df)
coords = list(zip(df['Географическая широта'], df['Географическая долгота']))

# ----------------------------
# 2. Создание полносвязного графа с edge_attr (расстояния)
# ----------------------------
edge_index = []
edge_attr = []

for i in range(N):
    for j in range(N):
        if i != j:
            edge_index.append([i, j])
            dist = haversine(coords[i], coords[j])
            edge_attr.append([dist])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ----------------------------
# 3. GNN Policy Network
# ----------------------------
class GNNPolicy(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_attr_dim=1):
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
# 4. Beam Search для генерации "учителя"
# ----------------------------
def beam_search_route(start, coords, speed_kmh=30.0, beam_size=5):
    """
    Генерирует хороший маршрут из start с помощью beam search.
    Возвращает маршрут (list) и total_time (в часах).
    """
    N = len(coords)
    # Каждый элемент в куче: (total_time, current_node, visited_tuple, route_list)
    heap = [(0.0, start, (start,), [start])]
    
    best_complete = None
    best_time = float('inf')

    while heap:
        time_so_far, current, visited_tuple, route = heappop(heap)
        visited_set = set(visited_tuple)

        if len(route) == N:
            if time_so_far < best_time:
                best_time = time_so_far
                best_complete = route
            continue

        # Ограничиваем размер луча
        if len(heap) > beam_size * N:
            continue

        for next_node in range(N):
            if next_node in visited_set:
                continue
            d_km = haversine(coords[current], coords[next_node])
            time_to_next = d_km / speed_kmh
            new_time = time_so_far + time_to_next
            new_route = route + [next_node]
            new_visited = visited_tuple + (next_node,)

            heappush(heap, (new_time, next_node, new_visited, new_route))

        # Оставляем только top-K по времени
        if len(heap) > beam_size:
            # Удаляем худшие (самые большие time) — но heap мин-куча, поэтому просто ограничим рост
            pass

    if best_complete is None:
        # fallback: nearest neighbor
        return nearest_neighbor_route(start, coords, speed_kmh)
    return best_complete, best_time

def nearest_neighbor_route(start, coords, speed_kmh=30.0):
    N = len(coords)
    visited = [False] * N
    route = [start]
    visited[start] = True
    current = start
    total_time = 0.0
    for _ in range(N - 1):
        best_next = -1
        best_dist = float('inf')
        for j in range(N):
            if not visited[j]:
                d = haversine(coords[current], coords[j])
                if d < best_dist:
                    best_dist = d
                    best_next = j
        route.append(best_next)
        visited[best_next] = True
        total_time += best_dist / speed_kmh
        current = best_next
    return route, total_time

# ----------------------------
# 5. Imitation Learning Loss
# ----------------------------
def imitation_loss(model, data, coords, num_samples=8, beam_size=5):
    N = data.x.size(0)
    node_embeddings = model.encode(data.x, data.edge_index, data.edge_attr)
    total_loss = 0.0
    num_steps = 0

    for _ in range(num_samples):
        start = random.randint(0, N - 1)
        expert_route, _ = beam_search_route(start, coords, beam_size=beam_size)

        # Предвычислим маски: для шага t маска = все узлы ДО t в маршруте
        for t in range(len(expert_route) - 1):
            next_expert = expert_route[t + 1]
            # Маска: True для посещённых (до шага t включительно), но нам нужна ~mask для непосещённых
            visited_mask = torch.zeros(N, dtype=torch.bool)
            visited_mask[expert_route[:t+1]] = True  # ✅ НЕ in-place: создаём новую маску

            logits = model.decoder(node_embeddings).squeeze(-1)
            logits = logits.masked_fill(visited_mask, -1e9)  # маскируем посещённые
            log_probs = F.log_softmax(logits, dim=0)
            total_loss += -log_probs[next_expert]
            num_steps += 1

    return total_loss / num_steps if num_steps > 0 else torch.tensor(0.0, requires_grad=True)

# ----------------------------
# 6. Обучение
# ----------------------------
device = torch.device('cpu')
model = GNNPolicy(in_dim=7, hidden_dim=32, out_dim=32, edge_attr_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("🚀 Начало обучения (Imitation Learning от Beam Search)...")
for epoch in range(500):
    loss = imitation_loss(model, data, coords, num_samples=4, beam_size=3)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 50 == 0:
        # Оценка маршрута от узла 0
        h = model.encode(data.x, data.edge_index, data.edge_attr)
        visited = torch.zeros(N, dtype=torch.bool)
        current = 0
        visited[current] = True
        total_time = 0.0
        for _ in range(N - 1):
            logits = model.decoder(h).squeeze(-1)
            logits = logits.masked_fill(visited, -1e9)
            next_node = logits.argmax().item()
            d_km = haversine(coords[current], coords[next_node])
            total_time += d_km / 30.0 * 60  # в минутах
            visited[next_node] = True
            current = next_node

        print(f"Epoch {epoch:3d} | Imitation Loss: {loss.item():.4f} | Eval Time: {total_time:6.1f} min")

print("\n✅ Обучение завершено.")

# ----------------------------
# 7. Финальный маршрут от узла 0
# ----------------------------
def get_route(model, data, coords, speed_kmh=30.0):
    N = data.x.size(0)
    h = model.encode(data.x, data.edge_index, data.edge_attr)
    visited = torch.zeros(N, dtype=torch.bool)
    current = 0
    visited[current] = True
    route = [current]
    total_time = 0.0

    for _ in range(N - 1):
        logits = model.decoder(h).squeeze(-1)
        logits = logits.masked_fill(visited, -1e9)
        next_node = logits.argmax().item()
        route.append(next_node)
        d_km = haversine(coords[current], coords[next_node])
        total_time += d_km / speed_kmh * 60
        visited[next_node] = True
        current = next_node

    return route, total_time

final_route, final_time = get_route(model, data, coords)
print(f"\n🏁 Финальный маршрут:")
print(" -> ".join(str(i + 1) for i in final_route))
print(f"⏱️ Итоговое время: {final_time:.1f} минут")