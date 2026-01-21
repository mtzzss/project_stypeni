import sqlite3
import os
import uuid
import cv2
from datetime import datetime
from .auth import make_hash
DB_NAME = "zones_analysis.db"
IMAGES_DIR = "images"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    conn = get_connection()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT,
            territory_name TEXT,
            upload_date TIMESTAMP,
            original_path TEXT,
            processed_path TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            category TEXT,
            subzone TEXT,
            area_pixels INTEGER,
            area_m2 REAL,
            FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
        )
    ''')
    c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
    conn.commit()
    conn.close()

def register_user(username, password):
    password_hash = make_hash(password)
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    password_hash = make_hash(password)
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ? AND password_hash = ?", (username, password_hash))
    user = c.fetchone()
    conn.close()
    return user[0] if user else None

def get_username_by_id(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    res = c.fetchone()
    conn.close()
    return res[0] if res else "unknown"
def create_session(user_id):
    token = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)", (token, user_id, datetime.now()))
        conn.commit()
    return token

def get_user_by_token(token):
    with get_connection() as conn:
        cursor = conn.execute("SELECT user_id FROM sessions WHERE token = ?", (token,))
        row = cursor.fetchone()
    return row[0] if row else None

def delete_session(token):
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()

def save_analysis_to_disk(user_id, img_orig_bgr, img_proc_bgr, filename_raw, territory, measurements):
    conn = get_connection()
    c = conn.cursor()
    
    try:
        username = get_username_by_id(user_id)
        user_dir = os.path.join(IMAGES_DIR, username)
        
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        unique_code = str(uuid.uuid4())
        clean_filename = filename_raw.replace(" ", "_")
        
        path_orig = os.path.join(user_dir, f"{unique_code}_{clean_filename}")
        path_proc = os.path.join(user_dir, f"{unique_code}_{clean_filename}_proc.jpg")
        
        cv2.imwrite(path_orig, img_orig_bgr)
        cv2.imwrite(path_proc, img_proc_bgr)
        
        upload_date = datetime.now()
        
        c.execute('''
            INSERT INTO snapshots 
            (user_id, filename, territory_name, upload_date, original_path, processed_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, filename_raw, territory, upload_date, path_orig, path_proc))
        
        snapshot_id = c.lastrowid
        
        for m in measurements:
            c.execute('''
                INSERT INTO measurements (snapshot_id, category, subzone, area_pixels, area_m2)
                VALUES (?, ?, ?, ?, ?)
            ''', (snapshot_id, m['category'], m['subzone'], m['area_pixels'], m['area_m2']))
            
        conn.commit()
        return True, "Анализ успешно сохранен в архив"
        
    except Exception as e:
        conn.rollback()
        return False, f"Ошибка сохранения: {str(e)}"
    finally:
        conn.close()

def get_snapshot_full_info(snapshot_id):
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT original_path, processed_path, territory_name, upload_date FROM snapshots WHERE id = ?", (snapshot_id,))
    snap_data = c.fetchone()
    
    if not snap_data:
        conn.close()
        return None

    orig_path, proc_path, territory, date = snap_data
    
    c.execute("SELECT category, subzone, area_pixels, area_m2 FROM measurements WHERE snapshot_id = ?", (snapshot_id,))
    rows = c.fetchall()
    
    measurements = []
    for r in rows:
        measurements.append({
            'category': r[0],
            'subzone': r[1],
            'area_pixels': r[2],
            'area_m2': r[3]
        })
    
    conn.close()
    
    def load_image_safe(path):
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None 

    img_orig = load_image_safe(orig_path)
    img_proc = load_image_safe(proc_path)
    
    error_msg = None
    if img_orig is None or img_proc is None:
        error_msg = "Файл не найден"

    return {
        'territory': territory,
        'date': date,
        'measurements': measurements,
        'image_orig': img_orig,
        'image_proc': img_proc,
        'error': error_msg
    }
def get_user_snapshots(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, territory_name, upload_date FROM snapshots WHERE user_id = ? ORDER BY upload_date DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows
def get_unique_territories(user_id):
    """Возвращает список уникальных названий территорий для фильтра"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT DISTINCT territory_name FROM snapshots WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

def get_territory_history(user_id, territory_name):
    """
    Возвращает данные для построения графика:
    Дата, Название зоны (например, 'Лес'), Площадь в м2
    """
    conn = get_connection()
    c = conn.cursor()
    
    query = '''
        SELECT s.upload_date, m.subzone, m.area_m2
        FROM snapshots s
        JOIN measurements m ON s.id = m.snapshot_id
        WHERE s.user_id = ? AND s.territory_name = ?
        ORDER BY s.upload_date ASC
    '''
    
    c.execute(query, (user_id, territory_name))
    rows = c.fetchall()
    conn.close()
    return rows
def get_snapshots_for_territory(user_id, territory_name):
    with get_connection() as conn:
        sql = "SELECT id, upload_date, original_path FROM snapshots WHERE user_id = ? AND territory_name = ? ORDER BY upload_date ASC"
        return conn.execute(sql, (user_id, territory_name)).fetchall()