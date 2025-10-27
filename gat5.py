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

# ----------------------------
# Путь к файлу
# ----------------------------
file_path = r"C:\Users\eliza\Downloads\test_list.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Файл не найден: {file_path}")

# ----------------------------
# 1. Загрузка и предобработка
# ----------------------------
df = pd.read_excel(file_path)[:50]  # Уменьшите до [:10] для точного TSP

def time_to_minutes(t):
    h, m = map(int, t.split(':'))
    return h * 60 + m

df['work_start'] = df['Время начала рабочего дня'].apply(time_to_minutes)
df['work_end']   = df['Время окончания рабочего дня'].apply(time_to_minutes)
df['lunch_start'] = df['Время начала обеда'].apply(time_to_minutes)
df['lunch_end']   = df['Время окончания обеда'].apply(time_to_minutes)
df['is_vip'] = (df['Уровень клиента'] == 'VIP').astype(int)

# Извлекаем временные окна как numpy массивы
work_start = df['work_start'].values  # в минутах
work_end   = df['work_end'].values

lunch_start = df['lunch_start'].values
lunch_end   = df['lunch_end'].values
is_vip = df['is_vip'].values  # уже есть в вашем коде

print(work_start)
print(work_end)

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

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

# ----------------------------
# 3. Создание графа
# ----------------------------
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ----------------------------
# 4. GNN Policy Network
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
# 5. REINFORCE с учётом временных окон
# ----------------------------
def reinforce_batch_step(model, data, coords, work_start, work_end, lunch_start, lunch_end, is_vip,
                        speed_kmh=30.0, num_samples=8, entropy_coef=0.01, time_window_penalty=1e6):
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

        current_time = 540.0
        start_time = current_time
        violated = False
        vip_count = 0

        # Учёт VIP для стартового узла
        if is_vip[current]:
            vip_count += 1

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

            current_time = arrival_time
            visited[next_node] = True
            current = next_node.item()

            # Считаем VIP
            if is_vip[next_node.item()]:
                vip_count += 1

        total_time_elapsed = current_time - start_time
        total_times.append(total_time_elapsed)

        base_reward = -current_time  # или -total_time_elapsed — решите, что логичнее
        reward = base_reward

        # 🔸 Бонус: +10% от модуля базовой награды за каждый VIP
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
# 6. Точный или жадный baseline (без учёта времени — только расстояние)
# ----------------------------
def compute_exact_tsp_time(coords, speed_kmh=30.0):
    N = len(coords)
    if N > 12:
        return compute_greedy_tsp_time(coords, speed_kmh)
    min_total_time = float('inf')
    for start in range(N):
        others = [i for i in range(N) if i != start]
        for perm in permutations(others):
            total_dist = sum(haversine(coords[perm[i]], coords[perm[i+1]]) for i in range(len(perm)-1))
            total_dist += haversine(coords[start], coords[perm[0]])  # если цикл, но у нас путь → убираем
            total_time = total_dist / speed_kmh * 60
            if total_time < min_total_time:
                min_total_time = total_time
    print(f"✅ Точное время (расстояние): {min_total_time:.1f} мин")
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
    print(f"🟢 Жадное время (расстояние): {min_time:.1f} мин")
    return min_time

# Baseline по расстоянию (без временных окон)
optimal_time = compute_greedy_tsp_time(coords, speed_kmh=30.0)

# ----------------------------
# 7. Обучение
# ----------------------------
device = torch.device('cpu')
model = GNNPolicy(in_dim=7, hidden_dim=32, out_dim=32, edge_attr_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
avg_times = []

print("🚀 Начало обучения с учётом временных окон...")
for epoch in range(1000):
    loss, avg_time = reinforce_batch_step(
        model, data, coords, work_start, work_end,
        speed_kmh=30.0, num_samples=2, time_window_penalty=10000, lunch_start=lunch_start, lunch_end=lunch_end, is_vip=is_vip
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
# 8. Финальный маршрут с учётом временных окон
# ----------------------------
def get_route(model, data, coords, work_start, work_end, lunch_start, lunch_end, is_vip, speed_kmh=30.0):
    N = data.x.size(0)
    visited = torch.zeros(N, dtype=torch.bool)
    current = 0
    visited[current] = True
    route = [current]
    current_time = 540.0
    start_time = current_time
    violated = False
    vip_count = int(is_vip[current])

    for _ in range(N - 1):
        h = model.encode(data.x, data.edge_index, data.edge_attr)
        mask = ~visited
        logits = model.decoder(h).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        next_node = logits.argmax().item()
        route.append(next_node)

        d_km = haversine(coords[current], coords[next_node])
        travel_time_min = d_km / speed_kmh * 60
        arrival_time = current_time + travel_time_min

        ws = work_start[next_node]
        we = work_end[next_node]
        ls = lunch_start[next_node]
        le = lunch_end[next_node]

        if arrival_time < ws or arrival_time > we or (ls <= arrival_time <= le):
            violated = True
        elif arrival_time < ws:
            arrival_time = ws

        current_time = arrival_time
        visited[next_node] = True
        current = next_node

        if is_vip[next_node]:
            vip_count += 1

    status = "❌ Нарушение окон" if violated else "✅ В рамках окон"
    print(f"   {status} | VIP посещено: {vip_count}")
    return route, current_time - start_time

def get_route_onnx(ort_session, data, coords, work_start, work_end, lunch_start, lunch_end, is_vip, speed_kmh=30.0):
    """
    Генерирует маршрут с использованием ONNX-модели (жадный выбор, как в get_route).
    
    Параметры:
    ----------
    ort_session : onnxruntime.InferenceSession
        Загруженная ONNX-модель.
    data : torch_geometric.data.Data
        Граф с x, edge_index, edge_attr.
    coords : list of (lat, lon)
        Координаты узлов.
    work_start, work_end, lunch_start, lunch_end, is_vip : np.ndarray
        Временные окна и признаки VIP.
    speed_kmh : float
        Скорость движения (км/ч).
        
    Возвращает:
    ----------
    route : list[int]
        Последовательность индексов узлов.
    total_time : float
        Общее время маршрута в минутах.
    """
    N = data.x.size(0)
    visited = np.zeros(N, dtype=bool)
    current = 0
    visited[current] = True
    route = [current]
    current_time = 540.0  # 09:00 утра в минутах
    start_time = current_time
    violated = False
    vip_count = int(is_vip[current])

    # Подготовка входов один раз (они не меняются в процессе маршрута)
    x_np = data.x.cpu().numpy().astype(np.float32)
    edge_index_np = data.edge_index.cpu().numpy().astype(np.int64)
    edge_attr_np = data.edge_attr.cpu().numpy().astype(np.float32)

    for _ in range(N - 1):
        # ONNX inference: получаем logits для всех узлов
        ort_inputs = {
            'x': x_np,
            'edge_index': edge_index_np,
            'edge_attr': edge_attr_np
        }
        logits = ort_session.run(None, ort_inputs)[0]  # shape: [N]

        # Применяем маску: запрещаем посещённые узлы
        mask = ~visited  # numpy bool array
        logits_masked = np.where(mask, logits, -1e9)

        # Жадный выбор
        next_node = int(np.argmax(logits_masked))
        route.append(next_node)

        # Расчёт времени прибытия
        d_km = haversine(coords[current], coords[next_node])
        travel_time_min = d_km / speed_kmh * 60
        arrival_time = current_time + travel_time_min

        ws = work_start[next_node]
        we = work_end[next_node]
        ls = lunch_start[next_node]
        le = lunch_end[next_node]

        # Проверка временных окон
        if arrival_time < ws or arrival_time > we or (ls <= arrival_time <= le):
            violated = True
        elif arrival_time < ws:
            arrival_time = ws  # Ждём открытия

        current_time = arrival_time
        visited[next_node] = True
        current = next_node

        if is_vip[next_node]:
            vip_count += 1

    status = "❌ Нарушение окон" if violated else "✅ В рамках окон"
    print(f"   {status} | VIP посещено: {vip_count}")
    return route, current_time - start_time

final_route, final_time = get_route(model, data, coords, work_start, work_end, lunch_start=lunch_start, lunch_end=lunch_end, is_vip=is_vip)
print(f"\n🏁 Финальный маршрут (после обучения):")
print(" -> ".join(str(i + 1) for i in final_route))
print(f"⏱️ Итоговое время с учётом окон: {final_time:.1f} минут")

# ----------------------------
# 9. Графики
# ----------------------------
epochs = list(range(1000))

plt.figure(figsize=(14, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, color='tab:red')
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Время маршрута
plt.subplot(1, 2, 2)
plt.plot(epochs, avg_times, color='tab:blue', label='GNN Policy (avg)')
plt.axhline(y=optimal_time, color='green', linestyle='--', linewidth=2, label=f'Greedy (dist only): {optimal_time:.1f} min')
plt.axhline(y=final_time, color='orange', linestyle='-.', linewidth=2, label=f'Final GNN Route: {final_time:.1f} min')
plt.title('Route Time (min) vs Epoch (with time windows)')
plt.xlabel('Epoch')
plt.ylabel('Time (minutes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


def export_model_to_onnx(model, data, filepath, opset_version=13):
    """
    Экспортирует модель GNNPolicy в формат ONNX.
    
    Параметры:
    ----------
    model : torch.nn.Module
        Обученная модель (GNNPolicy).
    data : torch_geometric.data.Data
        Пример данных для трассировки (должен содержать x, edge_index, edge_attr).
    filepath : str
        Путь для сохранения .onnx файла (например, 'model.onnx').
    opset_version : int
        Версия ONNX opset (рекомендуется 13+ для поддержки scatter и masked_fill).
    """
    model.eval()  # Переводим в режим оценки

    # Подготовка входов — фиксированные тензоры
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    # Обёртка для совместимости с ONNX (поскольку forward требует mask=None по умолчанию)
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, edge_index, edge_attr):
            # В inference мы не используем маску — она задаётся внутри маршрутизации
            logits = self.model(x, edge_index, edge_attr, mask=None)
            return logits

    wrapped_model = ONNXWrapper(model)

    # Экспорт
    torch.onnx.export(
        wrapped_model,
        (x, edge_index, edge_attr),
        filepath,
        input_names=['x', 'edge_index', 'edge_attr'],
        output_names=['logits'],
        dynamic_axes={
            'x': {0: 'num_nodes'},               # N может меняться
            'edge_index': {1: 'num_edges'},      # E может меняться
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

print("\n🏁 Маршрут с использованием ONNX-модели:")
onnx_route, onnx_time = get_route_onnx(
    ort_session, data, coords, work_start, work_end,
    lunch_start, lunch_end, is_vip, speed_kmh=30.0
)

print(" -> ".join(str(i + 1) for i in onnx_route))
print(f"⏱️ Время маршрута (ONNX): {onnx_time:.1f} минут")

# Сравнение с PyTorch-версией
print(f"\nСравнение:")
print(f"PyTorch: {final_time:.1f} мин")
print(f"ONNX   : {onnx_time:.1f} мин")
print("✅ Результаты совпадают!" if abs(final_time - onnx_time) < 1e-3 else "⚠️ Расхождение!")