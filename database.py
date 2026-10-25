"""
Database Gateway Module with PostgreSQL and SQLite Fallback Support
====================================================================
Designed for AI Powered Health Care System.

Behavior:
1. Detects PostgreSQL configuration via:
   - POSTGRES_KEY
   - DATABASE_URL
   - POSTGRES_URL
   - SUPABASE_DB_URL
   - Individual PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
2. Attempts to connect to PostgreSQL (production deployment).
3. If PostgreSQL is not configured, unreachable, or credentials fail:
   - Gracefully falls back to local SQLite database: 'health_gateway.db'.
   - System never crashes; operations proceed seamlessly.
4. Automatically migrates existing accounts from 'users_db.json' on first initialization.
"""

import os
import sys
import json
import time
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

# Load environment variables from .env or .env.local if present
def _load_local_env():
    for fname in [".env.local", ".env"]:
        p = Path(fname)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

_load_local_env()

SQLITE_PATH = Path("health_gateway.db")
USERS_JSON_PATH = Path("users_db.json")

class DatabaseManager:
    def __init__(self):
        self.backend: str = "sqlite"  # 'postgresql' or 'sqlite'
        self.is_fallback: bool = True
        self.connection_target: str = str(SQLITE_PATH)
        self.pg_driver = None
        self._lock = threading.Lock()
        self.pg_conn_str: Optional[str] = None
        
        # Check environment for PostgreSQL configuration
        self._detect_and_init_backend()

    def _detect_and_init_backend(self):
        """Determine whether to use PostgreSQL or SQLite."""
        pg_candidate = (
            os.getenv("POSTGRES_KEY") or
            os.getenv("DATABASE_URL") or
            os.getenv("POSTGRES_URL") or
            os.getenv("SUPABASE_DB_URL") or
            os.getenv("NEON_DATABASE_URL")
        )

        # Also support discrete PG credentials if provided
        if not pg_candidate and os.getenv("PGHOST") and (os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD")):
            host = os.getenv("PGHOST")
            port = os.getenv("PGPORT", "5432")
            user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER", "postgres")
            pw = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD", "")
            db = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB", "postgres")
            pg_candidate = f"postgresql://{user}:{pw}@{host}:{port}/{db}"

        # Clean up connection string if supplied
        if pg_candidate:
            pg_candidate = pg_candidate.strip()
            # Standardize postgres:// to postgresql:// for psycopg compatibility
            if pg_candidate.startswith("postgres://"):
                pg_candidate = "postgresql://" + pg_candidate[len("postgres://"):]
            self.pg_conn_str = pg_candidate

        # Attempt PostgreSQL if a connection string is present
        if self.pg_conn_str:
            pg_success = self._try_init_postgres()
            if pg_success:
                return

        # Fallback to SQLite
        self._init_sqlite()

    def _try_init_postgres(self) -> bool:
        """Attempt to load PostgreSQL driver and establish connection."""
        print("[*] PostgreSQL configuration detected. Attempting connection...")

        # Try psycopg2 first
        try:
            import psycopg2
            from psycopg2 import extras
            self.pg_driver = "psycopg2"
        except ImportError:
            try:
                import psycopg
                self.pg_driver = "psycopg"
            except ImportError:
                print("[!] Notice: Neither psycopg2 nor psycopg installed. To use PostgreSQL on deployment, add 'psycopg2-binary' to requirements.txt.")
                return False

        try:
            if self.pg_driver == "psycopg2":
                import psycopg2
                conn = psycopg2.connect(self.pg_conn_str, connect_timeout=5)
            else:
                import psycopg
                conn = psycopg.connect(self.pg_conn_str, timeout=5)

            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            conn.close()

            self.backend = "postgresql"
            self.is_fallback = False
            self.connection_target = self._mask_pg_url(self.pg_conn_str)
            print("=" * 70)
            print(f"[+] DATABASE BACKEND: PostgreSQL (PRODUCTION ACTIVE)")
            print(f"[+] Target: {self.connection_target}")
            print("=" * 70)

            self._create_pg_tables()
            self._migrate_json_users()
            return True

        except Exception as e:
            print("[!] PostgreSQL connection failed:")
            print(f"    Error: {e}")
            print("[!] Activating local SQLite fallback...")
            return False

    def _mask_pg_url(self, url: str) -> str:
        """Mask password in connection URL for safe logging."""
        try:
            if "@" in url and "://" in url:
                scheme, rest = url.split("://", 1)
                creds, host_part = rest.split("@", 1)
                user = creds.split(":", 1)[0]
                return f"{scheme}://{user}:****@{host_part.split('?')[0]}"
        except Exception:
            pass
        return "postgresql://[configured]"

    def _init_sqlite(self):
        """Initialize local SQLite database fallback."""
        self.backend = "sqlite"
        self.is_fallback = True
        self.connection_target = str(SQLITE_PATH)
        print("=" * 70)
        print(f"[!] DATABASE BACKEND: SQLite (LOCAL FALLBACK ACTIVE)")
        print(f"[!] Database file: {SQLITE_PATH.absolute()}")
        print(f"[i] PostgreSQL will automatically activate once POSTGRES_KEY / DATABASE_URL is set on deployment.")
        print("=" * 70)

        self._create_sqlite_tables()
        self._migrate_json_users()

    # -------------------------------------------------------------------------
    # Connection Helpers
    # -------------------------------------------------------------------------
    def _get_pg_conn(self):
        if self.pg_driver == "psycopg2":
            import psycopg2
            return psycopg2.connect(self.pg_conn_str)
        else:
            import psycopg
            return psycopg.connect(self.pg_conn_str)

    def _get_sqlite_conn(self):
        conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------------------------------------------------------
    # Schema Creation
    # -------------------------------------------------------------------------
    def _create_sqlite_tables(self):
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE,
                phone_number TEXT,
                age INTEGER DEFAULT 22,
                gender INTEGER DEFAULT 1,
                height_cm REAL DEFAULT 175.0,
                weight_kg REAL DEFAULT 70.0,
                bmi REAL DEFAULT 22.86,
                favorite_food TEXT DEFAULT '',
                password_hash TEXT NOT NULL,
                avatar_url TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_identifier TEXT,
                heart_rate REAL,
                spo2 REAL,
                temp_c REAL,
                predicted_condition TEXT,
                class_id INTEGER,
                confidence_scores TEXT,
                bmi REAL,
                treatment_mode TEXT,
                created_at TEXT
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_identifier TEXT,
                query TEXT,
                response TEXT,
                created_at TEXT
            );
            """)
            conn.commit()

    def _create_pg_tables(self):
        try:
            conn = self._get_pg_conn()
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    full_name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE,
                    phone_number VARCHAR(50),
                    age INTEGER DEFAULT 22,
                    gender INTEGER DEFAULT 1,
                    height_cm REAL DEFAULT 175.0,
                    weight_kg REAL DEFAULT 70.0,
                    bmi REAL DEFAULT 22.86,
                    favorite_food VARCHAR(255) DEFAULT '',
                    password_hash VARCHAR(255) NOT NULL,
                    avatar_url VARCHAR(500) DEFAULT '',
                    created_at VARCHAR(50),
                    updated_at VARCHAR(50)
                );
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    id SERIAL PRIMARY KEY,
                    user_identifier VARCHAR(255),
                    heart_rate REAL,
                    spo2 REAL,
                    temp_c REAL,
                    predicted_condition VARCHAR(100),
                    class_id INTEGER,
                    confidence_scores TEXT,
                    bmi REAL,
                    treatment_mode VARCHAR(50),
                    created_at VARCHAR(50)
                );
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS clinical_queries (
                    id SERIAL PRIMARY KEY,
                    user_identifier VARCHAR(255),
                    query TEXT,
                    response TEXT,
                    created_at VARCHAR(50)
                );
                """)
            conn.close()
        except Exception as e:
            print(f"[!] Error creating PostgreSQL tables: {e}. Falling back to SQLite.")
            self._init_sqlite()

    # -------------------------------------------------------------------------
    # Automatic Migration from legacy users_db.json
    # -------------------------------------------------------------------------
    def _migrate_json_users(self):
        if not USERS_JSON_PATH.exists():
            return
        try:
            with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or not data:
                return

            migrated = 0
            for k, u in data.items():
                email = (u.get("email") or ("" if "@" not in k else k)).strip().lower()
                phone = (u.get("phone_number") or ("" if "@" in k else k)).strip()
                ident = email or phone or k

                existing = self.get_user(ident)
                if not existing:
                    self.create_user({
                        "full_name": u.get("full_name", ident),
                        "email": email,
                        "phone_number": phone,
                        "age": u.get("age", 22),
                        "gender": u.get("gender", 1),
                        "height_cm": u.get("height_cm", 175.0),
                        "weight_kg": u.get("weight_kg", 70.0),
                        "bmi": u.get("bmi", 22.86),
                        "favorite_food": u.get("favorite_food", ""),
                        "password_hash": u.get("password_hash", ""),
                        "avatar_url": u.get("avatar_url", ""),
                        "created_at": u.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    migrated += 1
            if migrated > 0:
                print(f"[+] Migrated {migrated} existing user(s) from users_db.json into {self.backend.upper()}.")
        except Exception as e:
            print(f"[!] Note on JSON migration: {e}")

    # -------------------------------------------------------------------------
    # Status & Diagnostic Reporting
    # -------------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "is_fallback": self.is_fallback,
            "target": self.connection_target,
            "status": "active_healthy",
            "sqlite_file_exists": SQLITE_PATH.exists(),
            "pg_configured": bool(self.pg_conn_str)
        }

    # -------------------------------------------------------------------------
    # User CRUD Operations
    # -------------------------------------------------------------------------
    def get_user(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Look up user by email or phone number or ID."""
        if not identifier:
            return None
        ident_clean = identifier.strip().lower()

        if self.backend == "postgresql":
            try:
                conn = self._get_pg_conn()
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, full_name, email, phone_number, age, gender,
                               height_cm, weight_kg, bmi, favorite_food, password_hash,
                               avatar_url, created_at, updated_at
                        FROM users
                        WHERE LOWER(email) = %s OR phone_number = %s
                        LIMIT 1;
                    """, (ident_clean, identifier.strip()))
                    row = cur.fetchone()
                conn.close()
                if row:
                    cols = ["id", "full_name", "email", "phone_number", "age", "gender",
                            "height_cm", "weight_kg", "bmi", "favorite_food", "password_hash",
                            "avatar_url", "created_at", "updated_at"]
                    return dict(zip(cols, row))
                return None
            except Exception as e:
                print(f"[!] PostgreSQL query error: {e}. Attempting SQLite lookup fallback.")

        # SQLite query
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, full_name, email, phone_number, age, gender,
                       height_cm, weight_kg, bmi, favorite_food, password_hash,
                       avatar_url, created_at, updated_at
                FROM users
                WHERE LOWER(email) = ? OR phone_number = ?
                LIMIT 1;
            """, (ident_clean, identifier.strip()))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def create_user(self, u: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a newly registered user."""
        email_clean = (u.get("email") or "").strip().lower()
        phone_clean = (u.get("phone_number") or "").strip()
        created_at = u.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S")

        if self.backend == "postgresql":
            try:
                conn = self._get_pg_conn()
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO users (
                            full_name, email, phone_number, age, gender,
                            height_cm, weight_kg, bmi, favorite_food,
                            password_hash, avatar_url, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (
                        u.get("full_name", "").strip(),
                        email_clean or None,
                        phone_clean or None,
                        u.get("age", 22),
                        u.get("gender", 1),
                        u.get("height_cm", 175.0),
                        u.get("weight_kg", 70.0),
                        u.get("bmi", 22.86),
                        u.get("favorite_food", "").strip(),
                        u.get("password_hash", ""),
                        u.get("avatar_url", ""),
                        created_at,
                        created_at
                    ))
                    new_id = cur.fetchone()[0]
                conn.close()
                u["id"] = new_id
                return u
            except Exception as e:
                print(f"[!] PostgreSQL insert failed: {e}. Falling back to SQLite.")

        # SQLite insert
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (
                    full_name, email, phone_number, age, gender,
                    height_cm, weight_kg, bmi, favorite_food,
                    password_hash, avatar_url, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                u.get("full_name", "").strip(),
                email_clean or None,
                phone_clean or None,
                u.get("age", 22),
                u.get("gender", 1),
                u.get("height_cm", 175.0),
                u.get("weight_kg", 70.0),
                u.get("bmi", 22.86),
                u.get("favorite_food", "").strip(),
                u.get("password_hash", ""),
                u.get("avatar_url", ""),
                created_at,
                created_at
            ))
            conn.commit()
            u["id"] = cursor.lastrowid
            return u

    def update_user(self, identifier: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update user record fields."""
        user = self.get_user(identifier)
        if not user:
            return None

        user_id = user["id"]
        # Merge updates
        for k, v in updates.items():
            if v is not None:
                user[k] = v

        # Recalculate BMI if height or weight were modified
        if "height_cm" in updates or "weight_kg" in updates:
            h_m = float(user["height_cm"]) / 100.0
            user["bmi"] = round(float(user["weight_kg"]) / (h_m ** 2), 2)

        user["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        if self.backend == "postgresql":
            try:
                conn = self._get_pg_conn()
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE users SET
                            full_name = %s,
                            email = %s,
                            phone_number = %s,
                            age = %s,
                            gender = %s,
                            height_cm = %s,
                            weight_kg = %s,
                            bmi = %s,
                            favorite_food = %s,
                            password_hash = %s,
                            avatar_url = %s,
                            updated_at = %s
                        WHERE id = %s;
                    """, (
                        user["full_name"],
                        user["email"],
                        user["phone_number"],
                        user["age"],
                        user["gender"],
                        user["height_cm"],
                        user["weight_kg"],
                        user["bmi"],
                        user["favorite_food"],
                        user["password_hash"],
                        user["avatar_url"],
                        user["updated_at"],
                        user_id
                    ))
                conn.close()
                return user
            except Exception as e:
                print(f"[!] PostgreSQL update failed: {e}. Falling back to SQLite.")

        # SQLite update
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET
                    full_name = ?,
                    email = ?,
                    phone_number = ?,
                    age = ?,
                    gender = ?,
                    height_cm = ?,
                    weight_kg = ?,
                    bmi = ?,
                    favorite_food = ?,
                    password_hash = ?,
                    avatar_url = ?,
                    updated_at = ?
                WHERE id = ?;
            """, (
                user["full_name"],
                user["email"],
                user["phone_number"],
                user["age"],
                user["gender"],
                user["height_cm"],
                user["weight_kg"],
                user["bmi"],
                user["favorite_food"],
                user["password_hash"],
                user["avatar_url"],
                user["updated_at"],
                user_id
            ))
            conn.commit()
            return user

    def update_avatar(self, identifier: str, avatar_url: str) -> bool:
        user = self.get_user(identifier)
        if not user:
            return False
        return self.update_user(identifier, {"avatar_url": avatar_url}) is not None

    # -------------------------------------------------------------------------
    # Telemetry Log Persistence
    # -------------------------------------------------------------------------
    def save_telemetry(self, t: Dict[str, Any], user_identifier: Optional[str] = None) -> int:
        metrics = t.get("metrics", {})
        hr = metrics.get("hr", t.get("heart_rate", 72.0))
        spo2 = metrics.get("spo2", t.get("spo2", 98.0))
        temp = metrics.get("temp", t.get("temp_c", 36.6))
        cond = t.get("predicted_condition", metrics.get("condition", "Normal"))
        class_id = t.get("class_id", 0)
        conf_str = json.dumps(t.get("confidence_scores", []))
        bmi = metrics.get("bmi", 22.86)
        mode = t.get("treatment_mode", "both")
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")

        if self.backend == "postgresql":
            try:
                conn = self._get_pg_conn()
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO telemetry_logs (
                            user_identifier, heart_rate, spo2, temp_c,
                            predicted_condition, class_id, confidence_scores,
                            bmi, treatment_mode, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (user_identifier, hr, spo2, temp, cond, class_id, conf_str, bmi, mode, created_at))
                    tid = cur.fetchone()[0]
                conn.close()
                return tid
            except Exception as e:
                print(f"[!] PostgreSQL telemetry log failed: {e}. Falling back to SQLite.")

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry_logs (
                    user_identifier, heart_rate, spo2, temp_c,
                    predicted_condition, class_id, confidence_scores,
                    bmi, treatment_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (user_identifier, hr, spo2, temp, cond, class_id, conf_str, bmi, mode, created_at))
            conn.commit()
            return cursor.lastrowid

    # -------------------------------------------------------------------------
    # Clinical Query Log Persistence
    # -------------------------------------------------------------------------
    def save_query(self, user_identifier: Optional[str], query: str, response: str) -> int:
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")

        if self.backend == "postgresql":
            try:
                conn = self._get_pg_conn()
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO clinical_queries (user_identifier, query, response, created_at)
                        VALUES (%s, %s, %s, %s) RETURNING id;
                    """, (user_identifier, query, response, created_at))
                    qid = cur.fetchone()[0]
                conn.close()
                return qid
            except Exception as e:
                print(f"[!] PostgreSQL query log failed: {e}. Falling back to SQLite.")

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO clinical_queries (user_identifier, query, response, created_at)
                VALUES (?, ?, ?, ?);
            """, (user_identifier, query, response, created_at))
            conn.commit()
            return cursor.lastrowid

# Singleton instance for imports
db = DatabaseManager()

