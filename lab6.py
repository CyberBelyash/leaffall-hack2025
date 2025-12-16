import sys
import psycopg2
from psycopg2 import Error
from pathlib import Path

# === Конфигурация подключения к PostgreSQL ===
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "postgres",
    "user": "postgres",
    "password": "123456789"  # ← ЗАМЕНИТЕ НА СВОЙ ПАРОЛЬ!
}

# Поддерживаемые расширения (более двух типов: документы, изображения, видео — как в п. 6.6.2)
SUPPORTED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'txt', 'mp4'}


def get_connection():
    """Возвращает новое соединение с базой данных."""
    return psycopg2.connect(**DB_CONFIG)


# === 1. Инициализация БД: создание таблиц по Рисунку 1 ===
def init_db():
    """
    Создаёт таблицы:
      - parts (part_id SERIAL PK, part_name TEXT)
      - part_drawings (
            part_id INT PK REFERENCES parts ON DELETE CASCADE,
            file_extension TEXT CHECK (поддерживаемые расширения),
            drawing_data BYTEA NOT NULL,
            created_at TIMESTAMPTZ
        )
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # Таблица parts
            cur.execute("""
                CREATE TABLE IF NOT EXISTS parts (
                    part_id SERIAL PRIMARY KEY,
                    part_name TEXT NOT NULL
                );
            """)
            # Таблица part_drawings
            cur.execute("""
                CREATE TABLE IF NOT EXISTS part_drawings (
                    part_id INTEGER NOT NULL PRIMARY KEY
                        REFERENCES parts(part_id) ON DELETE CASCADE,
                    file_extension TEXT NOT NULL 
                        CHECK (file_extension IN ('png','jpg','jpeg','pdf','txt','mp4')),
                    drawing_data BYTEA NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            conn.commit()
        print("✅ Таблицы 'parts' и 'part_drawings' успешно созданы.")
    except Error as e:
        print(f"❌ Ошибка при инициализации БД: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


# === 2. Удаление таблиц (с каскадным удалением зависимостей) ===
def drop_table():
    """Удаляет обе таблицы. part_drawings удаляется автоматически при удалении parts (благодаря ON DELETE CASCADE),
       но удаляем явно для надёжности и чистоты."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS part_drawings;")
            cur.execute("DROP TABLE IF EXISTS parts;")
        conn.commit()
        print("✅ Таблицы 'parts' и 'part_drawings' удалены.")
    except Error as e:
        print(f"❌ Ошибка при удалении таблиц: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


# === 3. Добавление файла в БД как BLOB (drawing_data BYTEA) ===
def add_file(filepath: str):
    """
    Добавляет файл в БД:
      - создаёт новую запись в parts (part_name = имя файла);
      - создаёт новую запись в part_drawings (drawing_data = содержимое файла как BYTEA).
    Поддерживает: png, jpg, pdf, txt, mp4.
    """
    fp = Path(filepath)
    if not fp.exists():
        print(f"❌ Файл не найден: {filepath}")
        return

    ext = fp.suffix.lower().lstrip('.')
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"❌ Неподдерживаемое расширение: '{ext}'.\n"
              f"✅ Допустимые: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    try:
        with open(fp, 'rb') as f:
            blob_data = f.read()

        part_name = fp.name

        conn = get_connection()
        with conn.cursor() as cur:
            # ВСЕГДА создаём новую запись в parts
            cur.execute(
                "INSERT INTO parts (part_name) VALUES (%s) RETURNING part_id;",
                (part_name,)
            )
            part_id = cur.fetchone()[0]

            # ВСЕГДА создаём новую запись в part_drawings
            cur.execute(
                "INSERT INTO part_drawings (part_id, file_extension, drawing_data) "
                "VALUES (%s, %s, %s);",
                (part_id, ext, psycopg2.Binary(blob_data))
            )
            conn.commit()
            print(f"✅ Файл '{part_name}' сохранён в БД с part_id = {part_id}")

    except Exception as e:
        print(f"❌ Ошибка при добавлении файла: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


# === 4. Чтение и восстановление BLOB (документы, изображения, видео) ===
def get_file(part_id: int, output_dir: str):
    """
    Извлекает BLOB по part_id, сохраняет в output_dir.
    Автоматически добавляет расширение, если его нет в part_name.
    """
    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.part_name, pd.file_extension, pd.drawing_data
                FROM parts p
                JOIN part_drawings pd ON p.part_id = pd.part_id
                WHERE p.part_id = %s;
            """, (part_id,))
            row = cur.fetchone()

            if not row:
                print(f"❌ Запись с part_id = {part_id} не найдена.")
                return

            part_name, ext, data = row
            # Формируем имя файла: part_name + .ext (если расширение отсутствует)
            if not part_name.lower().endswith(f".{ext}"):
                filename = f"{part_name}.{ext}"
            else:
                filename = part_name

            full_path = out_path / filename
            with open(full_path, 'wb') as f:
                f.write(data)
            print(f"✅ Файл восстановлен: {full_path.absolute()}")

    except Exception as e:
        print(f"❌ Ошибка при извлечении файла: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


# === 5. Вывод списка всех файлов ===
def list_files():
    """Выводит таблицу: part_id, part_name, file_extension, created_at."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.part_id, p.part_name, pd.file_extension, pd.created_at
                FROM parts p
                JOIN part_drawings pd ON p.part_id = pd.part_id
                ORDER BY p.part_id;
            """)
            rows = cur.fetchall()

            if not rows:
                print("📭 В базе данных пока нет файлов.")
                return

            print(f"{'ID':<4} {'Имя файла':<30} {'Тип':<8} {'Дата'}")
            print("-" * 70)
            for pid, name, ext, dt in rows:
                print(f"{pid:<4} {name:<30} {ext:<8} {dt}")

    except Exception as e:
        print(f"❌ Ошибка при получении списка: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            


# === 6. Главное меню (интерактивный и пакетный режимы) ===
def print_help():
    print("\n📌 Доступные команды:")
    print("  init          — создать/пересоздать таблицы")
    print("  drop          — удалить таблицы")
    print("  add <путь>    — добавить файл (поддержка: png, jpg, pdf, txt, mp4)")
    print("  get <id> <папка> — извлечь файл по ID")
    print("  list          — показать список всех файлов")
    print("  help          — показать эту справку")
    print("  exit / quit   — завершить программу\n")

def delete(part_id):

    try:

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM parts AS p WHERE p.part_id = %s;
            """, (part_id,))
            
            print(f"✅ Файл с id = {part_id} успешно удалён")
            conn.commit()

    except Exception as e:
        print(f"❌ Ошибка при извлечении файла: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


def main():
    print("🔧 Лабораторная работа №6: Работа с BLOB в Python (PostgreSQL)")
    print("📌 Структура БД: parts (part_id, part_name) → part_drawings (part_id, file_extension, drawing_data BYTEA)")
    print("✅ Поддерживаемые форматы: png, jpg, pdf, txt, mp4 (более двух типов)")

    # Пакетный режим (python lab6.py add file.pdf)
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "init":
            init_db()
        elif cmd == "drop":
            drop_table()
        elif cmd == "add" and len(sys.argv) >= 3:
            add_file(sys.argv[2])
        elif cmd == "get" and len(sys.argv) >= 4:
            try:
                fid = int(sys.argv[2])
                get_file(fid, sys.argv[3])
            except ValueError:
                print("❌ ID должен быть целым числом.")
        elif cmd == "list":
            list_files()
        elif cmd == "del":
            delete(sys.argv[2])
        else:
            print_help()
        return

    # Интерактивный режим
    print_help()
    while True:
        try:
            user_input = input("lab6> ").strip()
            if not user_input:
                continue
            parts = user_input.split(maxsplit=2)
            cmd = parts[0].lower()

            if cmd in ("exit", "quit"):
                print("👋 Работа завершена.")
                break
            elif cmd == "help":
                print_help()
            elif cmd == "init":
                init_db()
            elif cmd == "drop":
                drop_table()
            elif cmd == "add":
                if len(parts) < 2:
                    print("❌ Укажите путь к файлу: add ./document.pdf")
                else:
                    add_file(parts[1])
            elif cmd == "get":
                if len(parts) < 3:
                    print("❌ Укажите ID и папку: get 1 ./output/")
                else:
                    try:
                        fid = int(parts[1])
                        get_file(fid, parts[2])
                    except ValueError:
                        print("❌ ID должен быть целым числом.")
            elif cmd == "list":
                list_files()
            elif cmd == "del":
                delete(parts[1])
            else:
                print(f"❌ Неизвестная команда: '{cmd}'. Введите 'help'.")

        except KeyboardInterrupt:
            print("\n⚠️  Прервано пользователем (Ctrl+C).")
            break
        except EOFError:
            print("\n👋 Выход.")
            break


if __name__ == "__main__":
    main()