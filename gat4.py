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
import random
import matplotlib.pyplot as plt
from itertools import permutations

# ----------------------------
# Путь к файлу
# ----------------------------
file_path = r"C:\Users\eliza\Downloads\test_list.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Файл не найден: {file_path}")

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

def reinforce_batch_step(model, data, coords, speed_kmh=30.0, num_samples=8, entropy_coef=0.01):
    N = data.x.size(0)
    
    # Предвычисляем эмбеддинги ОДИН РАЗ (с градиентами!)
    node_embeddings = model.encode(data.x, data.edge_index, data.edge_attr)

    total_times = []
    log_prob_sums = []
    entropy_sums = []

    for _ in range(num_samples):
        visited = torch.zeros(N, dtype=torch.bool)
        current = random.randint(0, N-1)
        visited[current] = True
        log_probs = []
        entropies = []
        total_time_hours = 0.0

        for step in range(N - 1):
            logits = model.decoder(node_embeddings).squeeze(-1)
            mask = ~visited
            logits = logits.masked_fill(~mask, -1e9)
            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
            next_node = dist.sample()

            log_probs.append(dist.log_prob(next_node))
            entropies.append(dist.entropy())

            d_km = haversine(coords[current], coords[next_node.item()])
            total_time_hours += d_km / speed_kmh

            visited[next_node] = True
            current = next_node.item()

        total_times.append(total_time_hours)
        log_prob_sums.append(torch.stack(log_probs).sum())
        entropy_sums.append(torch.stack(entropies).sum())

    total_times = torch.tensor(total_times, dtype=torch.float)
    log_prob_sums = torch.stack(log_prob_sums)
    entropy_sums = torch.stack(entropy_sums)

    rewards = -total_times
    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    policy_loss = -(advantage * log_prob_sums).mean()
    entropy_loss = -entropy_sums.mean()
    loss = policy_loss + entropy_coef * entropy_loss

    avg_time_min = total_times.mean().item() * 60
    return loss, avg_time_min

# ----------------------------
# 5. Вычисление оптимального или жадного времени маршрута
# ----------------------------
def compute_exact_tsp_time(coords, speed_kmh=30.0):
    N = len(coords)
    if N > 12:
        print(f"⚠️ N={N} > 12: полный перебор слишком дорог. Используем жадный алгоритм как приближение.")
        return compute_greedy_tsp_time(coords, speed_kmh)
    
    min_total_time = float('inf')
    for start in range(N):
        others = [i for i in range(N) if i != start]
        for perm in permutations(others):
            route = [start] + list(perm)
            total_dist = 0.0
            for i in range(len(route) - 1):
                total_dist += haversine(coords[route[i]], coords[route[i+1]])
            total_time = total_dist / speed_kmh * 60  # в минутах
            if total_time < min_total_time:
                min_total_time = total_time
    print(f"✅ Найден точный оптимальный маршрут за {min_total_time:.1f} мин (N={N})")
    return min_total_time

def compute_greedy_tsp_time(coords, speed_kmh=30.0):
    N = len(coords)
    min_time = float('inf')
    for start in range(N):
        visited = [False] * N
        current = start
        visited[current] = True
        total_dist = 0.0
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
            visited[nearest] = True
            current = nearest
        total_time = total_dist / speed_kmh * 60
        if total_time < min_time:
            min_time = total_time
    print(f"🟢 Использован жадный алгоритм (N={N} > 12). Приближённое время: {min_time:.1f} мин")
    return min_time

# Вычисляем baseline время
optimal_time = compute_exact_tsp_time(coords, speed_kmh=30.0)

# ----------------------------
# 6. Обучение с батчевым REINFORCE + сохранение метрик
# ----------------------------
device = torch.device('cpu')
model = GNNPolicy(in_dim=7, hidden_dim=32, out_dim=32, edge_attr_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
avg_times = []

print("🚀 Начало обучения (Batched REINFORCE с нормализацией advantage)...")
for epoch in range(1000):
    loss, avg_time = reinforce_batch_step(model, data, coords, speed_kmh=30.0, num_samples=2)
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
# Получение финального маршрута (детерминированный) — ДО графика!
# ----------------------------
def get_route(model, data, coords, speed_kmh=30.0):
    N = data.x.size(0)
    visited = torch.zeros(N, dtype=torch.bool)
    current = 0
    visited[current] = True
    route = [current]
    total_time = 0.0

    for _ in range(N - 1):
        h = model.encode(data.x, data.edge_index, data.edge_attr)
        mask = ~visited
        logits = model.decoder(h).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        next_node = logits.argmax().item()
        route.append(next_node)

        d_km = haversine(coords[current], coords[next_node])
        total_time += d_km / speed_kmh * 60  # в минутах

        visited[next_node] = True
        current = next_node

    return route, total_time

final_route, final_time = get_route(model, data, coords)
print(f"\n🏁 Финальный маршрут (после обучения):")
print(" -> ".join(str(i + 1) for i in final_route))
print(f"⏱️ Итоговое время: {final_time:.1f} минут")

# ----------------------------
# Построение графиков (только один график времени, с двумя горизонтальными линиями)
# ----------------------------
epochs = list(range(1000))

plt.figure(figsize=(14, 5))

# График Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, color='tab:red')
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# График времени — с зелёной И оранжевой линиями
plt.subplot(1, 2, 2)
plt.plot(epochs, avg_times, color='tab:blue', label='GNN Policy (avg during training)')
plt.axhline(y=optimal_time, color='green', linestyle='--', linewidth=2, label=f'Optimal / Greedy: {optimal_time:.1f} min')
plt.axhline(y=final_time, color='orange', linestyle='-.', linewidth=2, label=f'Final GNN Route: {final_time:.1f} min')
plt.title('Average Route Time (min) vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Time (minutes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
