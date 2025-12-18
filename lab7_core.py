# lab7_core.py
import psycopg2
import random
from datetime import datetime, timedelta
import time
import json  # ← критически важен для EXPLAIN JSON
from typing import Dict, Any, List
from contextlib import contextmanager


def get_db_config():
    from dotenv import load_dotenv
    import os
    load_dotenv()
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD"),
    }


def get_safe_db_params(config):
    cfg = config.copy()
    if "database" in cfg:
        cfg["dbname"] = cfg.pop("database")
    return cfg


def connect_admin():
    cfg = get_safe_db_params(get_db_config())
    conn = psycopg2.connect(**cfg)
    conn.autocommit = True
    return conn


def connect_lab():
    cfg = get_safe_db_params(get_db_config())
    cfg["dbname"] = "lab7_db"
    return psycopg2.connect(**cfg)


@contextmanager
def get_cursor(conn):
    """Контекстный менеджер для безопасной работы с курсором."""
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def setup_database(conn):
    with get_cursor(conn) as cur:
        cur.execute("SHOW server_version_num;")
        version_row = cur.fetchone()
        if version_row is None:
            raise RuntimeError("Failed to fetch PostgreSQL version")
        version = int(version_row[0])

        if version >= 130000:
            cur.execute("DROP DATABASE IF EXISTS lab7_db WITH (FORCE);")
        else:
            cur.execute("""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = 'lab7_db';
            """)
            cur.execute("DROP DATABASE IF EXISTS lab7_db;")
        cur.execute("CREATE DATABASE lab7_db;")


def create_schema(conn):
    with get_cursor(conn) as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS customers (id SERIAL PRIMARY KEY, name TEXT NOT NULL);")
        cur.execute("CREATE TABLE IF NOT EXISTS products (id SERIAL PRIMARY KEY, name TEXT NOT NULL, category TEXT);")

        customers_data = [(f"Customer_{i}",) for i in range(1, 1001)]
        products_data = [(f"Product_{i}", random.choice(['Electronics','Clothing','Books','Food'])) for i in range(1, 101)]
        cur.executemany("INSERT INTO customers (name) VALUES (%s);", customers_data)
        cur.executemany("INSERT INTO products (name, category) VALUES (%s, %s);", products_data)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                order_date TIMESTAMP NOT NULL,
                customer_id INT NOT NULL,
                product_id INT NOT NULL,
                quantity INT NOT NULL CHECK (quantity > 0),
                price NUMERIC(10, 2) NOT NULL CHECK (price >= 0)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS orders_partitioned (
                id SERIAL,
                order_date TIMESTAMP NOT NULL,
                customer_id INT NOT NULL,
                product_id INT NOT NULL,
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
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {name} PARTITION OF orders_partitioned
                FOR VALUES FROM ('{start}') TO ('{end}');
            """)
    conn.commit()


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


def insert_orders(conn, orders):
    batch_size = 5000
    insert_sql = "INSERT INTO {} (order_date, customer_id, product_id, quantity, price) VALUES (%s, %s, %s, %s, %s)"

    with get_cursor(conn) as cur:
        start = time.time()
        for i in range(0, len(orders), batch_size):
            cur.executemany(insert_sql.format("orders"), orders[i:i+batch_size])
        conn.commit()
        time_orders = time.time() - start

        start = time.time()
        for i in range(0, len(orders), batch_size):
            cur.executemany(insert_sql.format("orders_partitioned"), orders[i:i+batch_size])
        conn.commit()
        time_partitioned = time.time() - start

        cur.execute("SELECT COUNT(*) FROM orders")
        cnt1 = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM orders_partitioned")
        cnt2 = cur.fetchone()[0]

    return {
        "total_orders": len(orders),
        "inserted_orders": cnt1,
        "inserted_partitioned": cnt2,
        "time_orders_sec": round(time_orders, 2),
        "time_partitioned_sec": round(time_partitioned, 2)
    }


def benchmark_query(conn, table_name: str, date_from: str, date_to: str):
    sql = f"SELECT COUNT(*) FROM {table_name} WHERE order_date BETWEEN %s AND %s;"
    with get_cursor(conn) as cur:
        # Получаем план выполнения в формате JSON (возвращается как строка)
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}", (date_from, date_to))
        plan_row = cur.fetchone()
        if plan_row is None:
            raise RuntimeError("EXPLAIN returned no result")

        try:
            # Парсим JSON-строку → получаем список из одного элемента
            explain_data = json.loads(plan_row[0])
            plan = explain_data[0]  # первый и единственный объект плана
            exec_time = plan["Execution Time"]  # теперь работает!
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            raw = str(plan_row[0])[:250] + "..." if len(str(plan_row[0])) > 250 else str(plan_row[0])
            raise RuntimeError(f"Failed to parse EXPLAIN JSON: {e}, raw: {raw}")

        # Выполняем сам запрос для получения COUNT
        cur.execute(sql, (date_from, date_to))
        count_row = cur.fetchone()
        count = count_row[0] if count_row else 0

    return {"count": count, "exec_time_ms": round(exec_time, 2)}


def create_indexes(conn):
    with get_cursor(conn) as cur:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_date ON orders (order_date);")
        for part in ["q1_orders", "q2_orders", "q3_orders", "q4_orders"]:
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{part}_date ON {part} (order_date);")
    conn.commit()


def analyze_tables(conn):
    with get_cursor(conn) as cur:
        cur.execute("SET maintenance_work_mem = '256MB';")
        tables = ["orders", "q1_orders", "q2_orders", "q3_orders", "q4_orders"]
        start = time.time()
        for tbl in tables:
            cur.execute(f"ANALYZE {tbl};")
    return round(time.time() - start, 2)


def get_table_sizes(conn):
    with get_cursor(conn) as cur:
        cur.execute("""
            SELECT
                tablename,
                pg_size_pretty(pg_total_relation_size(tablename)) AS total_size,
                pg_size_pretty(pg_indexes_size(tablename)) AS index_size,
                pg_total_relation_size(tablename) AS total_bytes,
                pg_indexes_size(tablename) AS index_bytes
            FROM (VALUES ('orders'), ('q1_orders'), ('q2_orders'), ('q3_orders'), ('q4_orders')) AS t(tablename);
        """)
        rows = cur.fetchall()
        return [
            {
                "table": row[0],
                "total_size": row[1],
                "index_size": row[2],
                "total_bytes": row[3],
                "index_bytes": row[4]
            }
            for row in rows
        ]