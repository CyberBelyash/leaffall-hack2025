import random
import time
from datetime import datetime, timedelta, timezone
import psycopg2


DB_HOST = "10.241.173.110"
DB_PORT = 8294
DB_USER = "postgres"
DB_PASS = "yourpassword"  
DB_NAME = "test"


def connect(dbname="postgres"):
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        dbname=dbname
    )


def log(msg, status="info"):
    icon = "✅" if status == "ok" else "⚠️" if status == "warn" else "▶"
    print(f"{icon} {msg}")



def create_database():
    log("Создание БД 'test' (если не существует)...")
    conn = connect("postgres")
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
        if cur.fetchone():
            log("БД 'test' уже существует.", "ok")
        else:
            cur.execute(f'CREATE DATABASE "{DB_NAME}";')
            log("БД 'test' успешно создана.", "ok")
    finally:
        cur.close()
        conn.close()



def enable_timescaledb():
    log("Активация расширения timescaledb...")
    conn = connect(DB_NAME)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        log("Расширение timescaledb активировано.", "ok")
    finally:
        cur.close()
        conn.close()



def create_hypertable():
    log("Создание гипертаблицы 'sensors_data'...")
    conn = connect(DB_NAME)
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sensors_data (
                time TIMESTAMPTZ NOT NULL,
                sensor_id INT NOT NULL,
                value FLOAT NOT NULL
            );
        """)
        cur.execute("SELECT create_hypertable('sensors_data', 'time', if_not_exists => TRUE);")
        conn.commit()
        log("Гипертаблица 'sensors_data' создана.", "ok")
    finally:
        cur.close()
        conn.close()



def insert_data(n_records=100_000):
    log(f"Генерация и вставка {n_records:,} записей...")
    conn = connect(DB_NAME)
    cur = conn.cursor()
    
    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    batch_size = 5000
    total_inserted = 0

    try:
        for offset in range(0, n_records, batch_size):
            batch = []
            for i in range(offset, min(offset + batch_size, n_records)):
                ts = start_time + timedelta(seconds=10 * i)
                sensor_id = random.randint(1, 10)
                value = round(random.uniform(-10, 10), 3)
                batch.append((ts, sensor_id, value))

            cur.executemany(
                "INSERT INTO sensors_data (time, sensor_id, value) VALUES (%s, %s, %s)",
                batch
            )
            conn.commit()
            total_inserted += len(batch)
            print(f"  → {total_inserted:,}/{n_records:,}", end="\r", flush=True)
            time.sleep(0.01)
        print(f"\n✅ Вставлено {total_inserted:,} записей.")
    finally:
        cur.close()
        conn.close()



def analyze_query():
    log("Выполнение анализа запроса (среднее за последний час)...")
    conn = connect(DB_NAME)
    cur = conn.cursor()
    try:
        
        cur.execute("""
            SELECT sensor_id, AVG(value) AS avg_val
            FROM sensors_data
            WHERE time > NOW() - INTERVAL '1 hour'
            GROUP BY sensor_id;
        """)
        rows = cur.fetchall()
        if rows:
            for sensor_id, avg_val in rows:
                print(f"  → Датчик {sensor_id}: {avg_val:.4f}")
        else:
            print("  → Нет данных за последний час (нормально — данные старше).")

        
        print("\n📊 EXPLAIN ANALYZE:")
        cur.execute("""
            EXPLAIN (ANALYZE, BUFFERS)
            SELECT sensor_id, AVG(value)
            FROM sensors_data
            WHERE time > NOW() - INTERVAL '1 hour'
            GROUP BY sensor_id;
        """)
        for row in cur.fetchall():
            print("   ", row[0].strip())
    finally:
        cur.close()
        conn.close()



def create_continuous_agg():
    log("Создание continuous aggregate 'sensors_hourly_avg'...")
    conn = connect(DB_NAME)
    conn.autocommit = True  
    cur = conn.cursor()
    try:
        
        cur.execute("DROP MATERIALIZED VIEW IF EXISTS sensors_hourly_avg;")

       
        cur.execute("""
            CREATE MATERIALIZED VIEW sensors_hourly_avg
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', time) AS bucket,
                sensor_id,
                AVG(value) AS avg_value,
                COUNT(*) AS cnt
            FROM sensors_data
            GROUP BY bucket, sensor_id
            WITH NO DATA;
        """)

        
        cur.execute("""
            SELECT add_continuous_aggregate_policy('sensors_hourly_avg',
                start_offset => INTERVAL '1 month',
                end_offset => INTERVAL '1 hour',
                schedule_interval => INTERVAL '15 minutes');
        """)

        
        cur.execute("CALL refresh_continuous_aggregate('sensors_hourly_avg', NULL, NULL);")

        
        cur.execute("SELECT COUNT(*) FROM sensors_hourly_avg;")
        count = cur.fetchone()[0]
        log(f"Continuous aggregate создан. Записей: {count:,}", "ok")
    except Exception as e:
        log(f"Ошибка при создании continuous aggregate: {e}", "warn")
    finally:
        cur.close()
        conn.close()



def show_chunks_sample():
    log("Получение информации о чанках и выборка данных...")
    conn = connect(DB_NAME)
    cur = conn.cursor()

    try:
        
        cur.execute("""
            SELECT
                chunk_name,
                chunk_schema,
                range_start,
                range_end,
                pg_size_pretty(pg_total_relation_size(format('%I.%I', chunk_schema, chunk_name)::regclass)) AS size
            FROM timescaledb_information.chunks
            WHERE hypertable_name = 'sensors_data'
            ORDER BY range_start DESC
            LIMIT 3;
        """)
        chunks = cur.fetchall()

        if not chunks:
            print("  → Чанки не найдены (возможно, данные ещё не распределены).")
            return

        print(f"\n📦 Найдено чанков: {len(chunks)} (показаны самые свежие)")
        print("-" * 80)

        for i, (chunk_name, chunk_schema, range_start, range_end, size) in enumerate(chunks, 1):
            full_table_name = f'"{chunk_schema}"."{chunk_name}"'
            start_str = range_start.strftime('%Y-%m-%d %H:%M') if range_start else str(range_start)
            end_str = range_end.strftime('%Y-%m-%d %H:%M') if range_end else str(range_end)
            print(f"Чанк #{i}: {chunk_name} ({full_table_name})")
            print(f"  Диапазон: {start_str} → {end_str} | Размер: {size}")

            
            try:
                cur.execute(f"""
                    SELECT time, sensor_id, value
                    FROM {full_table_name}
                    ORDER BY time ASC
                    LIMIT 3;
                """)
                rows = cur.fetchall()
                if rows:
                    print("  Пример данных:")
                    for j, (ts, sid, val) in enumerate(rows, 1):
                        ts_fmt = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                        print(f"    {j}. {ts_fmt} | sensor_id={sid} | value={val}")
                else:
                    print("  → Нет данных в чанке.")
            except Exception as e:
                print(f"  ⚠️ Ошибка выборки: {e}")

            print("-" * 80)

    except Exception as e:
        log(f"Ошибка при работе с чанками: {e}", "warn")
    finally:
        cur.close()
        conn.close()


def _print_chunk_rows(rows):
    if rows:
        print("  Пример данных:")
        for j, (ts, sid, val) in enumerate(rows, 1):
            ts_fmt = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
            print(f"    {j}. {ts_fmt} | sensor_id={sid} | value={val}")
    else:
        print("  → Нет данных в чанке.")



if __name__ == "__main__":
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА 8: TimescaleDB (порт 8294)")
    print("=" * 60)

    # Проверка подключения к PostgreSQL
    try:
        conn = connect("postgres")
        cur = conn.cursor()
        cur.execute("SELECT version();")
        ver = cur.fetchone()[0].split()[1]
        log(f"Подключено к PostgreSQL {ver}", "ok")
        cur.close()
        conn.close()
    except Exception as e:
        print("❌ Ошибка подключения. Убедитесь, что контейнер запущен:")
        print("   docker run -d --name timescaledb -p 8294:5432 \\")
        print("     -e POSTGRES_PASSWORD=yourpassword timescale/timescaledb:latest-pg14")
        exit(1)

    
    create_database()
    enable_timescaledb()
    create_hypertable()
    insert_data(n_records=100_000)
    analyze_query()
    create_continuous_agg()
    show_chunks_sample()  