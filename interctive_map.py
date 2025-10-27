import requests
import folium
from folium import Marker

def get_osrm_route_segment(coords, start_idx, end_idx, profile="walking"):
    """
    Получает геометрию и расстояние одного сегмента маршрута.
    
    :return: (geometry: List[(lat, lon)], distance_km: float)
    """
    start_lat, start_lon = coords[start_idx]
    end_lat, end_lon = coords[end_idx]
    url = f"http://router.project-osrm.org/route/v1/{profile}/{start_lon},{start_lat};{end_lon},{end_lat}?geometries=geojson&overview=full&annotations=true"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"OSRM segment request failed: {resp.status_code}")
    data = resp.json()
    route = data["routes"][0]
    geometry = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
    distance_m = route["distance"]  # в метрах
    distance_km = distance_m / 1000.0
    return geometry, distance_km


def visualize_route_on_map(coords, path, profile="walking", output_file="route_map.html"):
    """
    Создаёт интерактивную карту с сегментами маршрута и tooltip'ами длины.
    """
    if len(path) < 2:
        raise ValueError("Маршрут должен содержать хотя бы две точки.")

    # Центрируем карту на первой точке
    start_lat, start_lon = coords[path[0]]
    m = folium.Map(location=[start_lat, start_lon], zoom_start=14)

    # Добавляем маркеры всех точек
    for idx, (lat, lon) in enumerate(coords):
        color = "red" if idx in path else "blue"
        Marker(
            location=[lat, lon],
            popup=f"Point {idx}",
            icon=folium.Icon(color=color)
        ).add_to(m)

    # Нумерация посещения
    for order, idx in enumerate(path):
        Marker(
            location=coords[idx],
            icon=folium.DivIcon(html=f"""<div style="color: white; font-weight: bold; 
                                          background: red; border-radius: 50%; 
                                          width: 24px; height: 24px; 
                                          display: flex; align-items: center; 
                                          justify-content: center;">{order}</div>""")
        ).add_to(m)

    # Обрабатываем каждый сегмент маршрута
    for i in range(len(path) - 1):
        start_idx = path[i]
        end_idx = path[i + 1]

        try:
            geometry, dist_km = get_osrm_route_segment(coords, start_idx, end_idx, profile)
            tooltip_text = f"Сегмент {i} → {i+1}<br>Расстояние: {dist_km:.3f} км"
            folium.PolyLine(
                locations=geometry,
                color="green",
                weight=5,
                opacity=0.8,
                tooltip=tooltip_text
            ).add_to(m)
        except Exception as e:
            print(f"⚠️ Ошибка при получении сегмента {start_idx} → {end_idx}: {e}")

    m.save(output_file)
    print(f"✅ Карта с интерактивными сегментами сохранена в: {output_file}")


# === Пример использования ===
if __name__ == "__main__":
    coords = [
        (55.751244, 37.618423),  # Красная площадь
        (55.755826, 37.617299),  # ГУМ
        (55.749689, 37.620598),  # Храм Василия Блаженного
        (55.760085, 37.614970),  # Театральная площадь
    ]
    path = [0, 2, 1, 3]
    visualize_route_on_map(coords, path, profile="walking", output_file="route_with_distances.html")