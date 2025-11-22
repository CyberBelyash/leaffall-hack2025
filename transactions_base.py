import psycopg2
from psycopg2 import Error, extensions
from decimal import Decimal
import sys
import random

DB_CONFIG = {
    "database": "postgres",
    "user": "postgres",
    "password": "123456789",
    "host": "localhost",
    "port": "5432"
}

def maybe_raise():
    """С вероятностью 50% вызывает исключение — имитация сбоя."""
    if random.random() < 0.5:
        raise RuntimeError("❌ Случайный сбой: сетевой разрыв / таймаут / deadlock")

def init_database(connection):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account (
                    id BIGINT PRIMARY KEY,
                    balance NUMERIC(15, 2) NOT NULL CHECK (balance >= 0)
                );
            """)
            cursor.execute("SELECT COUNT(*) FROM account;")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO account (id, balance) VALUES (%s, %s), (%s, %s);",
                               (624001562408, Decimal('10000.00'), 2236781258763, Decimal('5000.00')))
            connection.commit()
    except Error as e:
        connection.rollback()
        print(f"❌ Ошибка инициализации БД: {e}")

def print_balances(connection):
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT id, balance FROM account ORDER BY id;")
            rows = cur.fetchall()
            print("Текущие балансы:")
            for r in rows:
                print(f"  ID {r[0]}: {r[1]:.2f}")
    except Error as e:
        print(f"❌ Ошибка при выводе балансов: {e}")

def transfer_funds(connection, from_id, to_id, amount):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT balance FROM account WHERE id = %s;", (from_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Счёт {from_id} не найден")
            balance_from = row[0]
            if balance_from < amount:
                raise ValueError(f"Недостаточно средств на счёте {from_id}")

            cursor.execute("SELECT balance FROM account WHERE id = %s;", (to_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Счёт {to_id} не найден")
            balance_to = row[0]

            # 🔥 Случайный сбой с 50% шансом
            maybe_raise()

            new_balance_from = balance_from - amount
            new_balance_to = balance_to + amount

            cursor.execute("UPDATE account SET balance = %s WHERE id = %s;", (new_balance_from, from_id))
            cursor.execute("UPDATE account SET balance = %s WHERE id = %s;", (new_balance_to, to_id))

            connection.commit()
            print("✅ Перевод выполнен. Обновлённые данные:")
            print_balances(connection)
            return True

    except (Error, ValueError, RuntimeError) as e:
        connection.rollback()
        print(f"❌ Транзакция откачена. Причина: {e}")
        return False

def adjust_balance(connection, account_id, delta):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT balance FROM account WHERE id = %s;", (account_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Счёт {account_id} не найден")
            old_balance = row[0]
            new_balance = old_balance + delta
            if new_balance < 0:
                raise ValueError(f"Операция приведёт к отрицательному балансу ({new_balance}). Запрещено.")

            # 🔥 Случайный сбой с 50% шансом
            maybe_raise()

            cursor.execute("UPDATE account SET balance = %s WHERE id = %s;", (new_balance, account_id))
            connection.commit()
            print(f"✅ Баланс счёта {account_id} изменён на {delta:+.2f}. Новый баланс: {new_balance:.2f}")
            print_balances(connection)
            return True

    except (Error, ValueError, RuntimeError) as e:
        connection.rollback()
        print(f"❌ Изменение баланса откачено. Причина: {e}")
        return False

def demo_savepoint(connection):
    try:
        with connection.cursor() as cursor:
            
            cursor.execute("SAVEPOINT sp1;")
            cursor.execute("UPDATE account SET balance = balance * 1.1;")
            cursor.execute("RELEASE SAVEPOINT sp1;")
            
            cursor.execute("SAVEPOINT sp2;")
            cursor.execute("SELECT")
            cursor.execute("ROLLBACK TO SAVEPOINT sp2;")

            print("  ➤ Балансы увеличены на 10%.")
            print_balances(connection)

            # 🔥 Случайный сбой с 50% шансом — имитируем ошибку после SAVEPOINT
            maybe_raise()

            # Если исключения не было — фиксируем
            connection.commit()
            print("✅ SAVEPOINT-сценарий: сбой не произошёл. Обновление зафиксировано.")
            print_balances(connection)

    except RuntimeError as e:
        # Откатываем только к SAVEPOINT
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK TO SAVEPOINT sp1;")
        connection.commit()
        print("🔄 Сбой в SAVEPOINT-сценарии. Откат к точке сохранения выполнен.")
        print("✅ Транзакция успешно завершена после отката.")
        print_balances(connection)

    except Error as e:
        connection.rollback()
        print(f"❌ Критическая ошибка в SAVEPOINT-сценарии: {e}")

def isolated_transfer_demo(connection, level_code, level_name):
    try:
        old_level = connection.isolation_level
        connection.set_isolation_level(level_code)
        with connection.cursor() as cursor:
            cursor.execute("UPDATE account SET balance = balance + 100 WHERE id = %s;", (624001562408,))

            # 🔥 Случайный сбой с 50% шансом
            maybe_raise()

            connection.commit()
            print(f"✅ Транзакция с уровнем изоляции '{level_name}' завершена. Обновление:")
            print_balances(connection)
    except (Error, RuntimeError) as e:
        connection.rollback()
        print(f"❌ Транзакция с уровнем '{level_name}' откачена. Причина: {e}")
    finally:
        connection.set_isolation_level(old_level)

def main():
    connection = None
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        connection.autocommit = False
        init_database(connection)

        print("ℹ️  Во всех транзакциях включён режим случайных сбоев (50% шанс).")
        print("   Это демонстрирует откат при исключениях, как требует п. 5.5.2.")

        while True:
            print("\n" + "="*60)
            print("1. Перевод средств")
            print("2. Изменить баланс (пополнить/списать)")
            print("3. Демонстрация SAVEPOINT (сбои → откат к точке)")
            print("4. READ COMMITTED (+100 к счёту)")
            print("5. REPEATABLE READ (+100 к счёту)")
            print("6. SERIALIZABLE (+100 к счёту)")
            print("7. Показать балансы")
            print("0. Выйти")
            choice = input("Ваш выбор: ").strip()

            if choice == "1":
                try:
                    aid1 = int(input("ID отправителя (по умолчанию 624001562408): ") or "624001562408")
                    aid2 = int(input("ID получателя (по умолчанию 2236781258763): ") or "2236781258763")
                    amt = Decimal(input("Сумма перевода (по умолчанию 2500): ") or "2500")
                    transfer_funds(connection, aid1, aid2, amt)
                except Exception as e:
                    print(f"❌ Ошибка ввода: {e}")

            elif choice == "2":
                try:
                    aid = int(input("ID счёта: ") or "624001562408")
                    delta = Decimal(input("Изменение баланса (например, +2000 или -500.75): "))
                    adjust_balance(connection, aid, delta)
                except Exception as e:
                    print(f"❌ Ошибка ввода: {e}")

            elif choice == "3":
                demo_savepoint(connection)

            elif choice == "4":
                isolated_transfer_demo(connection, extensions.ISOLATION_LEVEL_READ_COMMITTED, "READ COMMITTED")
            elif choice == "5":
                isolated_transfer_demo(connection, extensions.ISOLATION_LEVEL_REPEATABLE_READ, "REPEATABLE READ")
            elif choice == "6":
                isolated_transfer_demo(connection, extensions.ISOLATION_LEVEL_SERIALIZABLE, "SERIALIZABLE")

            elif choice == "7":
                print_balances(connection)

            elif choice == "0":
                break

            else:
                print("⚠️ Неверный выбор.")

    except Error as e:
        print(f"❌ Критическая ошибка подключения к БД: {e}")
        sys.exit(1)
    finally:
        if connection:
            connection.close()
            print("🔌 Соединение закрыто.")

if __name__ == "__main__":
    main()