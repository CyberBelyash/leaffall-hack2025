import psycopg2
import random
from datetime import datetime, timedelta
import os

# === Дополнительные импорты для визуализации ===
import matplotlib.pyplot as plt
import networkx as nx

# === Конфигурация подключения ===
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "postgres",
    "user": "postgres",
    "password": "123456789"
}

REPORT_FILE = r"lab7_report.txt"


def get_safe_db_params(config):
    cfg = config.copy()
    if "database" in cfg:
        cfg["dbname"] = cfg.pop("database")
    return cfg


def log_to_report(msg):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def execute_sql(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


def fetch_one_sql(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def fetch_all_sql(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def execute_and_time_sql(conn, sql_stmt):
    """
    Выполняет SQL-команду и возвращает время выполнения (в миллисекундах),
    используя clock_timestamp() на стороне PostgreSQL.
    sql_stmt — строка полной команды (например, 'INSERT INTO ...').
    """
    timing_sql = f"""
        DO $$
        DECLARE
            start_ts TIMESTAMP;
            end_ts TIMESTAMP;
            elapsed_ms DOUBLE PRECISION;
        BEGIN
            start_ts := clock_timestamp();
            {sql_stmt};
            end_ts := clock_timestamp();
            elapsed_ms := EXTRACT(EPOCH FROM (end_ts - start_ts)) * 1000;
            RAISE NOTICE 'elapsed_ms: %', elapsed_ms;
        END $$;
    """
    with conn.cursor() as cur:
        cur.execute("SET client_min_messages = NOTICE;")
        cur.execute(timing_sql)
        conn.commit()

        # Извлекаем NOTICE
        notices = conn.notices
        conn.notices.clear()
        for notice in reversed(notices):
            if "elapsed_ms:" in notice:
                try:
                    return float(notice.split("elapsed_ms:")[1].strip())
                except:
                    pass
    return 0.0


def sql_escape_value(val):
    """
    Безопасное представление Python-значения в SQL-литерал.
    Поддерживаем: None, int, float, str, datetime.
    """
    if val is None:
        return 'NULL'
    elif isinstance(val, bool):
        return 'TRUE' if val else 'FALSE'
    elif isinstance(val, (int, float)):
        return str(val)
    elif isinstance(val, str):
        # Экранируем одинарные кавычки
        return "'" + val.replace("'", "''") + "'"
    elif isinstance(val, datetime):
        return f"'{val.isoformat()}'"
    else:
        raise TypeError(f"Unsupported type for SQL: {type(val)}")


def setup_database(conn):
    log_to_report("\n[1] Подготовка базы данных...")
    version = fetch_one_sql(conn, "SHOW server_version_num;")[0]
    if int(version) >= 130000:
        drop_sql = "DROP DATABASE IF EXISTS lab7_db WITH (FORCE);"
    else:
        execute_sql(conn, """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = 'lab7_db';
        """)
        drop_sql = "DROP DATABASE IF EXISTS lab7_db;"
    execute_sql(conn, drop_sql)
    execute_sql(conn, "CREATE DATABASE lab7_db;")
    log_to_report("✅ БД lab7_db создана.")


def connect_to_lab_db():
    cfg = get_safe_db_params(DB_CONFIG)
    cfg["dbname"] = "lab7_db"
    return psycopg2.connect(**cfg)


def create_schema(conn):
    log_to_report("\n[2] Создание схемы (customers, products, orders, partitioned orders)...")
    
    execute_sql(conn, """
        CREATE TABLE IF NOT EXISTS customers (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL
        );
    """)
    
    execute_sql(conn, """
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT
        );
    """)
    
    customers_data = [(f"Customer_{i}",) for i in range(1, 1001)]
    products_data = [(f"Product_{i}", random.choice(['Electronics','Clothing','Books','Food'])) for i in range(1, 101)]
    
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO customers (name) VALUES (%s);", customers_data)
        cur.executemany("INSERT INTO products (name, category) VALUES (%s, %s);", products_data)
    conn.commit()
    log_to_report("✅ customers (1000), products (100) созданы и заполнены.")

    execute_sql(conn, """
        CREATE TABLE IF NOT EXISTS orders (
            id SERIAL PRIMARY KEY,
            order_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            customer_id INT NOT NULL REFERENCES customers(id),
            product_id INT NOT NULL REFERENCES products(id),
            quantity INT NOT NULL CHECK (quantity > 0),
            price NUMERIC(10, 2) NOT NULL CHECK (price >= 0)
        );
    """)

    execute_sql(conn, """
        CREATE TABLE IF NOT EXISTS orders_partitioned (
            id SERIAL,
            order_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            customer_id INT NOT NULL REFERENCES customers(id),
            product_id INT NOT NULL REFERENCES products(id),
            quantity INT NOT NULL CHECK (quantity > 0),
            price NUMERIC(10, 2) NOT NULL CHECK (price >= 0)
        ) PARTITION BY RANGE (order_date);
    """)

    partitions = [
        ("q1_orders", "2023-01-01", "2023-04-01"),
        ("q2_orders", "2023-04-01", "2023-07-01"),
        ("q3_orders", "2023-07-01", "2023-10-01"),
        ("q4_orders", "2023-10-01", "2024-01-01"),
    ]
    for name, start, end in partitions:
        execute_sql(conn, f"""
            CREATE TABLE IF NOT EXISTS {name} PARTITION OF orders_partitioned
            FOR VALUES FROM ('{start}') TO ('{end}');
        """)
    log_to_report("✅ orders и orders_partitioned (4 партиции) созданы.")


def generate_orders_data():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    current = start_date
    orders = []
    while current <= end_date:
        daily_count = random.randint(25, 40)
        for _ in range(daily_count):
            orders.append((
                current,
                random.randint(1, 1000),
                random.randint(1, 100),
                random.randint(1, 10),
                round(random.uniform(5.0, 200.0), 2)
            ))
        current += timedelta(days=1)
    return orders


def insert_orders(conn):
    log_to_report("\n[3] Вставка данных в orders и orders_partitioned...")
    orders = generate_orders_data()
    log_to_report(f"🔁 Генерация {len(orders)} заказов завершена.")

    # Преобразуем заказы в SQL-литералы
    rows_sql = []
    for row in orders:
        # row = (datetime, int, int, int, float)
        vals = [
            sql_escape_value(row[0]),  # order_date
            sql_escape_value(row[1]),  # customer_id
            sql_escape_value(row[2]),  # product_id
            sql_escape_value(row[3]),  # quantity
            sql_escape_value(row[4]),  # price
        ]
        rows_sql.append(f"({', '.join(vals)})")

    values_clause = ", ".join(rows_sql)

    # Вставка в orders
    insert_orders_sql = f"""
        INSERT INTO orders (order_date, customer_id, product_id, quantity, price)
        VALUES {values_clause}
    """
    time_orders = execute_and_time_sql(conn, insert_orders_sql)

    # Вставка в orders_partitioned
    insert_partitioned_sql = f"""
        INSERT INTO orders_partitioned (order_date, customer_id, product_id, quantity, price)
        VALUES {values_clause}
    """
    time_partitioned = execute_and_time_sql(conn, insert_partitioned_sql)

    cnt1 = fetch_one_sql(conn, "SELECT COUNT(*) FROM orders")[0]
    cnt2 = fetch_one_sql(conn, "SELECT COUNT(*) FROM orders_partitioned")[0]
    log_to_report(f"✅ Вставлено: {cnt1} в orders (за {time_orders:.2f} мс), {cnt2} в orders_partitioned (за {time_partitioned:.2f} мс)")


def benchmark_queries(conn):
    log_to_report("\n[4/6] Сравнение производительности (EXPLAIN ANALYZE)...")
    
    queries = [
        ("SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2023-04-01' AND '2023-06-30';", "обычная таблица"),
        ("SELECT COUNT(*) FROM orders_partitioned WHERE order_date BETWEEN '2023-04-01' AND '2023-06-30';", "партиционированная"),
    ]

    for sql, label in queries:
        plan = fetch_one_sql(conn, f"EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, FORMAT JSON) {sql}")[0][0]
        exec_time = plan['Execution Time']
        cnt = fetch_one_sql(conn, sql)[0]
        log_to_report(f"🔍 Запрос на {label}: {cnt} строк, время = {exec_time:.2f} мс")

    # Проверка Partition Pruning
    prune_rows = fetch_all_sql(conn, """
        EXPLAIN SELECT COUNT(*) FROM orders_partitioned 
        WHERE order_date BETWEEN '2023-04-01' AND '2023-06-30';
    """)
    prune_text = "\n".join(row[0] for row in prune_rows)
    if all(p not in prune_text for p in ["q1_orders", "q3_orders", "q4_orders"]):
        log_to_report("✅ Partition pruning работает: сканируется только q2_orders")
    else:
        log_to_report("⚠️ Partition pruning НЕ работает корректно")


def create_indexes(conn):
    log_to_report("\n[5] Создание индексов...")
    execute_sql(conn, "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders (order_date);")
    for part in ["q1_orders", "q2_orders", "q3_orders", "q4_orders"]:
        execute_sql(conn, f"CREATE INDEX IF NOT EXISTS idx_{part}_date ON {part} (order_date);")
    log_to_report("✅ Индексы по order_date созданы (1 для orders, 4 для партиций)")


def maintenance(conn):
    log_to_report("\n[7] Техническое обслуживание (ANALYZE)...")
    
    cfg = get_safe_db_params(DB_CONFIG)
    cfg["dbname"] = "lab7_db"
    maint_conn = psycopg2.connect(**cfg)
    maint_conn.autocommit = True

    try:
        execute_sql(maint_conn, "SET maintenance_work_mem = '256MB';")
        stmts = "; ".join(f"ANALYZE {tbl}" for tbl in ["orders", "q1_orders", "q2_orders", "q3_orders", "q4_orders"])
        duration_ms = execute_and_time_sql(maint_conn, stmts)
        log_to_report(f"✅ ANALYZE выполнен за {duration_ms:.2f} мс")

    except Exception as e:
        log_to_report(f"⚠️ Ошибка при ANALYZE: {e}")
        raise
    finally:
        maint_conn.close()


def final_stats(conn):
    log_to_report("\n[8] Итоговая статистика:")
    sizes = fetch_all_sql(conn, """
        SELECT
            schemaname || '.' || tablename AS table,
            pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size,
            pg_size_pretty(pg_indexes_size(schemaname || '.' || tablename)) AS index_size
        FROM pg_tables
        WHERE tablename = ANY(ARRAY['orders', 'q1_orders', 'q2_orders', 'q3_orders', 'q4_orders']);
    """)
    for row in sizes:
        log_to_report(f"📊 {row[0]:<20} — общий: {row[1]:<8}, индексы: {row[2]}")


def visualize_partitions(conn):
    log_to_report("\n[9] Визуализация партиций — создание графа...")
    
    partitions_info = fetch_all_sql(conn, """
        SELECT
            c.relname AS partition_name,
            pg_get_expr(c.relpartbound, c.oid) AS partition_expr
        FROM pg_class c
        JOIN pg_inherits i ON c.oid = i.inhrelid
        JOIN pg_class p ON p.oid = i.inhparent
        WHERE p.relname = 'orders_partitioned'
        ORDER BY c.relname;
    """)

    partition_counts = {}
    for name, _ in partitions_info:
        cnt = fetch_one_sql(conn, f"SELECT COUNT(*) FROM {name};")[0]
        partition_counts[name] = cnt

    import re
    partition_ranges = {}
    for name, expr in partitions_info:
        m = re.search(r"FROM $'([^']+)'.* TO $'([^']+)'", expr)
        if m:
            start, end = m.groups()
            start_date = start.split()[0]
            end_date = end.split()[0]
            partition_ranges[name] = (start_date, end_date)
        else:
            partition_ranges[name] = ("?", "?")

    G = nx.DiGraph()
    root = "orders_partitioned"
    G.add_node(root, label="root", shape="box", size=2000)

    pos = {root: (0, 0)}
    y_step = -1
    for i, (name, (start, end)) in enumerate(partition_ranges.items()):
        cnt = partition_counts.get(name, 0)
        G.add_node(name, shape="ellipse", size=1000)
        G.add_edge(root, name)
        pos[name] = (-2 + i * 1.5, y_step)

    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=[root], node_color='lightblue', node_size=3000, node_shape='s')
    nx.draw_networkx_nodes(G, pos, nodelist=list(partition_ranges.keys()), node_color='lightgreen', node_size=2500, node_shape='o')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')

    nx.draw_networkx_labels(G, pos, {root: root}, font_size=12, font_weight='bold')
    for name, (x, y) in pos.items():
        if name != root:
            cnt = partition_counts.get(name, 0)
            start, end = partition_ranges[name]
            label = f"{name}\n[{start} – {end}]\n{cnt:,} rows"
            plt.text(x, y - 0.2, label, fontsize=10, ha='center', va='center', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    plt.title("Граф партиций таблицы orders_partitioned", fontsize=14, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("lab7_partitions_graph.png", dpi=150)
    plt.show()
    log_to_report("✅ Граф партиций сохранён в lab7_partitions_graph.png и отображён.")


def run_lab():
    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)
    log_to_report("=" * 60)
    log_to_report("ЛАБОРАТОРНАЯ РАБОТА №7: PostgreSQL Table Partitioning (замер через clock_timestamp())")
    log_to_report("=" * 60)

    try:
        admin_conn = psycopg2.connect(**get_safe_db_params(DB_CONFIG))
        admin_conn.autocommit = True
        setup_database(admin_conn)
        admin_conn.close()

        conn = connect_to_lab_db()
        create_schema(conn)
        insert_orders(conn)
        benchmark_queries(conn)
        create_indexes(conn)
        benchmark_queries(conn)
        maintenance(conn)
        final_stats(conn)
        visualize_partitions(conn)
        conn.close()

        log_to_report("\n✅ Лабораторная работа завершена. Результаты сохранены в " + REPORT_FILE)

    except Exception as e:
        log_to_report(f"❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    run_lab()