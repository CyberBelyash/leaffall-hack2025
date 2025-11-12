# create_admin.py
import psycopg2
from passlib.context import CryptContext
import sys

# Конфигурация (берём из вашего FastAPI)
DB_CONFIG = {
    "database": "postgres",
    "user": "postgres",
    "password": "123456789",
    "host": "localhost",
    "port": "5432"
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def ensure_is_admin_column():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute("""
            ALTER TABLE users 
            ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE
        """)
        conn.commit()
        print("✅ Колонка `is_admin` добавлена (если её не было).")
    except Exception as e:
        print(f"⚠️ Ошибка при добавлении колонки is_admin: {e}")
    finally:
        cur.close()
        conn.close()

def create_admin(email: str, name: str, password: str):
    ensure_is_admin_column()
    
    hashed_pw = get_password_hash(password)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO users (email, name, hashed_password, is_admin)
            VALUES (%s, %s, %s, TRUE)
            ON CONFLICT (email) DO UPDATE
            SET 
                name = EXCLUDED.name,
                hashed_password = EXCLUDED.hashed_password,
                is_admin = TRUE
            RETURNING id, email;
        """, (email, name, hashed_pw))
        row = cur.fetchone()
        conn.commit()
        print(f"✅ Администратор создан/обновлён: ID={row[0]}, email={row[1]}")
    except Exception as e:
        conn.rollback()
        print(f"❌ Ошибка: {e}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Использование: python create_admin.py <email> <name> <password>")
        print("Пример: python create_admin.py admin@example.com 'Админ' admin123")
        sys.exit(1)
    
    email, name, password = sys.argv[1], sys.argv[2], sys.argv[3]
    create_admin(email, name, password)
    print("\nТеперь вы можете:")
    print("1. Войти на http://127.0.0.1:8080/auth.html с этими учётными данными")
    print("2. Перейти на http://127.0.0.1:8080/admin_panel.html")