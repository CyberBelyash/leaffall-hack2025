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
    
    # 🔥 Предвычисляем эмбеддинги ОДИН РАЗ (с градиентами!)
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
    entropy_loss = -entropy_sums.mean()  # максимизируем энтропию → минимизируем -H
    loss = policy_loss + entropy_coef * entropy_loss

    avg_time_min = total_times.mean().item() * 60
    return loss, avg_time_min

import matplotlib.pyplot as plt

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
# 7. Построение графиков
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

# График времени
plt.subplot(1, 2, 2)
plt.plot(epochs, avg_times, color='tab:blue')
plt.title('Average Route Time (min) vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Time (minutes)')
plt.grid(True)

plt.tight_layout()
plt.show()