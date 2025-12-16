import psycopg2
import random
from datetime import datetime, timedelta
import time
import os

# === Конфигурация подключения ===
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "postgres",  # ← база для администрирования
    "user": "postgres",
    "password": "123456789"  # ← ЗАМЕНИТЕ НА СВОЙ ПАРОЛЬ!
}

REPORT_FILE = r"lab7_report.txt"


def get_safe_db_params(config):
    """Преобразует 'database' → 'dbname' для psycopg2."""
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


def setup_database(conn):
    log_to_report("\n[1] Подготовка базы данных...")
    # Проверим версию PostgreSQL для совместимости с DROP DATABASE ... WITH (FORCE)
    version = fetch_one_sql(conn, "SHOW server_version_num;")[0]
    if int(version) >= 130000:
        drop_sql = "DROP DATABASE IF EXISTS lab7_db WITH (FORCE);"
    else:
        # Для старых версий: закрываем все соединения вручную
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
                current,  # ← order_date = current
                random.randint(1, 1000),   # customer_id
                random.randint(1, 100),    # product_id
                random.randint(1, 10),     # quantity
                round(random.uniform(5.0, 200.0), 2)  # price
            ))
        current += timedelta(days=1)
    return orders


def insert_orders(conn):
    log_to_report("\n[3] Вставка данных в orders и orders_partitioned...")
    orders = generate_orders_data()
    log_to_report(f"🔁 Генерация {len(orders)} заказов завершена.")

    batch_size = 5000
    insert_sql = """
        INSERT INTO {} (order_date, customer_id, product_id, quantity, price)
        VALUES (%s, %s, %s, %s, %s)
    """

    start = time.time()
    with conn.cursor() as cur:
        for i in range(0, len(orders), batch_size):
            cur.executemany(insert_sql.format("orders"), orders[i:i+batch_size])
    conn.commit()
    time_orders = time.time() - start

    start = time.time()
    with conn.cursor() as cur:
        for i in range(0, len(orders), batch_size):
            cur.executemany(insert_sql.format("orders_partitioned"), orders[i:i+batch_size])
    conn.commit()
    time_partitioned = time.time() - start

    cnt1 = fetch_one_sql(conn, "SELECT COUNT(*) FROM orders")[0]
    cnt2 = fetch_one_sql(conn, "SELECT COUNT(*) FROM orders_partitioned")[0]
    log_to_report(f"✅ Вставлено: {cnt1} в orders (за {time_orders:.2f} с), {cnt2} в orders_partitioned (за {time_partitioned:.2f} с)")


def benchmark_queries(conn):
    log_to_report("\n[4/6] Сравнение производительности (EXPLAIN ANALYZE)...")
    
    queries = [
        ("SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2023-04-01' AND '2023-06-30';", "обычная таблица"),
        ("SELECT COUNT(*) FROM orders_partitioned WHERE order_date BETWEEN '2023-04-01' AND '2023-06-30';", "партиционированная"),
    ]

    for sql, label in queries:
        plan = fetch_one_sql(conn, f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")[0][0]
        exec_time = plan['Execution Time']
        cnt = fetch_one_sql(conn, sql)[0]
        log_to_report(f"🔍 Запрос на {label}: {cnt} строк, время = {exec_time:.2f} мс")

    # Проверка Partition Pruning
    prune_text = fetch_one_sql(conn, """
        EXPLAIN SELECT COUNT(*) FROM orders_partitioned 
        WHERE order_date BETWEEN '2023-04-01' AND '2023-06-30';
    """)[0]
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
    log_to_report("\n[7] Техническое обслуживание (VACUUM ANALYZE, REINDEX)...")
    
    # Отдельное подключение с autocommit=True (обязательно для VACUUM и REINDEX)
    cfg = get_safe_db_params(DB_CONFIG)
    cfg["dbname"] = "lab7_db"
    maint_conn = psycopg2.connect(**cfg)
    maint_conn.autocommit = True

    try:
        tables = ["orders", "q1_orders", "q2_orders", "q3_orders", "q4_orders"]
        with maint_conn.cursor() as cur:
            for tbl in tables:
                cur.execute(f"REINDEX TABLE {tbl};")
                cur.execute(f"VACUUM ANALYZE {tbl};")
        log_to_report("✅ REINDEX и VACUUM ANALYZE выполнены")
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


def run_lab():
    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)
    log_to_report("="*60)
    log_to_report("ЛАБОРАТОРНАЯ РАБОТА №7: PostgreSQL Table Partitioning")
    log_to_report("="*60)

    try:
        admin_conn = psycopg2.connect(**get_safe_db_params(DB_CONFIG))
        admin_conn.autocommit = True
        setup_database(admin_conn)
        admin_conn.close()

        conn = connect_to_lab_db()
        create_schema(conn)
        insert_orders(conn)
        benchmark_queries(conn)         # ← до индексов
        create_indexes(conn)
        benchmark_queries(conn)         # ← после индексов
        maintenance(conn)
        final_stats(conn)
        conn.close()

        log_to_report("\n✅ Лабораторная работа завершена. Результаты сохранены в " + REPORT_FILE)

    except Exception as e:
        log_to_report(f"❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    run_lab()