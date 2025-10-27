import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch.distributions import Categorical
import pandas as pd
from haversine import haversine
import numpy as np
import os

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
            dist = haversine(coords[i], coords[j])  # в километрах
            edge_attr.append([dist])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # [E, 1]

# ----------------------------
# 3. Создание графа
# ----------------------------
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ----------------------------
# 4. GNN Policy Network с поддержкой edge_attr
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
# 5. Функция шага обучения (REINFORCE с baseline)
# ----------------------------
def reinforce_step(model, data, coords, baseline, speed_kmh=30.0, alpha=0.9):
    N = data.x.size(0)
    visited = torch.zeros(N, dtype=torch.bool)
    current = 0
    visited[current] = True
    log_probs = []
    total_time_hours = 0.0

    for step in range(N - 1):
        h = model.encode(data.x, data.edge_index, data.edge_attr)
        mask = ~visited
        logits = model.decoder(h).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits, dim=0)
        dist = Categorical(probs)
        next_node = dist.sample()
        log_probs.append(dist.log_prob(next_node))

        d_km = haversine(coords[current], coords[next_node.item()])
        total_time_hours += d_km / speed_kmh
        visited[next_node] = True
        current = next_node.item()

    # Обновляем baseline: экспоненциальное скользящее среднее
    new_baseline = alpha * baseline + (1 - alpha) * total_time_hours
    advantage = -(total_time_hours - baseline)  # отрицательное, если маршрут лучше baseline

    loss = -advantage * torch.stack(log_probs).sum()
    return loss, total_time_hours * 60, new_baseline  # возвращаем время в минутах и обновлённый baseline

# ----------------------------
# 6. Обучение с baseline
# ----------------------------
device = torch.device('cpu')
model = GNNPolicy(in_dim=7, hidden_dim=32, out_dim=32, edge_attr_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Инициализация baseline (например, как оценка "среднего" времени)
# Грубая оценка: среднее расстояние * N / скорость
avg_dist = edge_attr.mean().item()
initial_baseline_hours = (avg_dist * N) / 30.0
baseline = initial_baseline_hours

print("🚀 Начало обучения (REINFORCE с baseline и edge_attr)...")
for epoch in range(100):
    loss, total_time, baseline = reinforce_step(
        model, data, coords, baseline=baseline, speed_kmh=30.0, alpha=0.95
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():8.2f} | Total Time: {total_time:6.1f} min | Baseline: {baseline*60:6.1f} min")

print("\n✅ Обучение завершено.")

# ----------------------------
# 7. Получение финального маршрута (детерминированный)
# ----------------------------
def get_route(model, data, coords, speed_kmh=30.0):
    N = data.x.size(0)
    visited = torch.zeros(N, dtype=torch.bool)
    current = 0
    visited[current] = True
    route = [current]
    total_time = 0.0  # в минутах

    for _ in range(N - 1):
        h = model.encode(data.x, data.edge_index, data.edge_attr)
        mask = ~visited
        logits = model.decoder(h).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        next_node = logits.argmax().item()
        route.append(next_node)

        d_km = haversine(coords[current], coords[next_node])
        total_time += d_km / speed_kmh * 60  # сразу в минутах

        visited[next_node] = True
        current = next_node

    return route, total_time

final_route, final_time = get_route(model, data, coords)
print(f"\n🏁 Финальный маршрут (после обучения):")
print(" -> ".join(str(i + 1) for i in final_route))
print(f"⏱️ Итоговое время: {final_time:.1f} минут")