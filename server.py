import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recommendations_db import HEALTH_RECOMMENDATIONS
from database import db

# Ensure avatar and static storage exists
STATIC_DIR = Path("static")
AVATARS_DIR = STATIC_DIR / "avatars"
AVATARS_DIR.mkdir(parents=True, exist_ok=True)

USERS_DB_FILE = Path("users_db.json")

# =============================================================================
# Password Security (bcrypt with 12 salt rounds, with secure PBKDF2 fallback)
# =============================================================================
try:
    import bcrypt
    def hash_password(password: str) -> str:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    print("[+] Using native bcrypt (12 salt rounds) for authentication.")
except ImportError:
    import hashlib
    import secrets
    def hash_password(password: str) -> str:
        salt = secrets.token_hex(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return f"pbkdf2:12:{salt}:{key.hex()}"

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        if hashed_password.startswith("pbkdf2:12:"):
            _, _, salt, key = hashed_password.split(":")
            check = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 100000)
            return check.hex() == key
        return False
    print("[!] bcrypt module not found; using secure PBKDF2 fallback.")

def load_users() -> dict:
    if USERS_DB_FILE.exists():
        try:
            with open(USERS_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_users(users: dict):
    with open(USERS_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

# =============================================================================
# FastAPI Application & Middleware
# =============================================================================
app = FastAPI(title="AI-Enabled WBAN Health Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount /static for uploaded patient profile images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained XGBoost model
try:
    model = joblib.load("wban_xgboost_model.pkl")
    print("[+] XGBoost model loaded successfully.")
except Exception as e:
    print(f"[!] Warning: Could not load model: {e}. Run train_model.py first.")
    model = None

# =============================================================================
# Request Schemas
# =============================================================================
class TelemetryInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: int = Field(..., ge=0, le=1)  # 0: Female, 1: Male
    height_cm: float = Field(..., gt=50, lt=250)
    weight_kg: float = Field(..., gt=10, lt=300)
    heart_rate: float
    spo2: float
    temp_c: float
    treatment_mode: str = "both"  # "natural", "medical", or "both"

class RegisterInput(BaseModel):
    full_name: str
    email: str
    phone_number: Optional[str] = ""
    age: int = Field(..., ge=1, le=120)
    gender: int = Field(..., ge=0, le=1)  # 0: Female, 1: Male
    height_cm: float = Field(..., gt=50, lt=250)
    weight_kg: float = Field(..., gt=10, lt=300)
    favorite_food: Optional[str] = ""
    password: str = Field(..., min_length=4)

class LoginInput(BaseModel):
    identifier: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    password: str

class UpdateProfileInput(BaseModel):
    identifier: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    favorite_food: Optional[str] = None
    new_password: Optional[str] = None

class QueryInput(BaseModel):
    query: str
    telemetry: Optional[dict] = None

# Store most recent telemetry in memory for dashboard polling
latest_telemetry = {}

def find_user_key(users: dict, identifier: str) -> Optional[str]:
    ident = identifier.strip().lower()
    if ident in users:
        return ident
    for k, u in users.items():
        if k.lower() == ident:
            return k
        if u.get("email", "").strip().lower() == ident:
            return k
        if u.get("phone_number", "").strip() == identifier.strip():
            return k
    return None

def to_safe_user(u: dict) -> dict:
    return {
        "full_name": u.get("full_name", ""),
        "email": u.get("email", ""),
        "phone_number": u.get("phone_number", ""),
        "age": u.get("age", 22),
        "gender": u.get("gender", 1),
        "height_cm": u.get("height_cm", 175.0),
        "weight_kg": u.get("weight_kg", 70.0),
        "bmi": u.get("bmi", 22.86),
        "favorite_food": u.get("favorite_food", ""),
        "avatar_url": u.get("avatar_url", ""),
        "created_at": u.get("created_at", "")
    }

# =============================================================================
# User Account & Profile APIs
# =============================================================================
@app.get("/api/db_status")
def get_database_status():
    """Return active database status (PostgreSQL or SQLite fallback)."""
    return {
        "status": "success",
        "database": db.get_status()
    }

@app.post("/api/register")
def register_user(data: RegisterInput):
    email_clean = data.email.strip().lower()
    phone_clean = data.phone_number.strip() if data.phone_number else ""

    # Check if user already exists in DB or JSON
    existing = db.get_user(email_clean) or (db.get_user(phone_clean) if phone_clean else None)
    if not existing:
        users = load_users()
        existing_key = find_user_key(users, email_clean) or (find_user_key(users, phone_clean) if phone_clean else None)
        if existing_key:
            existing = users[existing_key]

    if existing:
        raise HTTPException(status_code=400, detail="Account already exists with this email or phone. Please sign in.")

    height_m = data.height_cm / 100.0
    bmi = round(data.weight_kg / (height_m ** 2), 2)
    hashed = hash_password(data.password)

    user_record = {
        "full_name": data.full_name.strip(),
        "email": email_clean,
        "phone_number": phone_clean,
        "age": data.age,
        "gender": data.gender,
        "height_cm": data.height_cm,
        "weight_kg": data.weight_kg,
        "bmi": bmi,
        "favorite_food": (data.favorite_food or "").strip(),
        "password_hash": hashed,
        "avatar_url": "",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Persist in active DB (PostgreSQL or SQLite fallback)
    created_user = db.create_user(user_record)

    # Sync into legacy JSON backup
    users = load_users()
    key = email_clean or phone_clean
    users[key] = created_user
    save_users(users)

    return {
        "status": "success",
        "message": f"Account created in {db.backend.upper()} with bcrypt security.",
        "database_backend": db.backend,
        "user": to_safe_user(created_user)
    }

@app.post("/api/login")
def login_user(data: LoginInput):
    ident = (data.identifier or data.email or data.phone_number or "").strip()
    if not ident:
        raise HTTPException(status_code=400, detail="Please enter your email or phone number.")

    user = db.get_user(ident)
    if not user:
        # Check JSON fallback
        users = load_users()
        user_key = find_user_key(users, ident)
        if user_key:
            user = users[user_key]
            # Sync user into database
            try:
                db.create_user(user)
            except Exception:
                pass

    if not user:
        raise HTTPException(status_code=404, detail="No registered account found with that email or phone.")

    if not verify_password(data.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Incorrect password. Authentication failed.")

    return {
        "status": "success",
        "message": "Authentication successful.",
        "database_backend": db.backend,
        "user": to_safe_user(user)
    }

@app.post("/api/update_profile")
def update_profile(data: UpdateProfileInput):
    ident = data.identifier.strip()
    user = db.get_user(ident)
    if not user:
        users = load_users()
        user_key = find_user_key(users, ident)
        if user_key:
            user = users[user_key]
            try:
                db.create_user(user)
            except Exception:
                pass

    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")

    updates = {}
    if data.full_name is not None:
        updates["full_name"] = data.full_name.strip()
    if data.email is not None:
        updates["email"] = data.email.strip().lower()
    if data.phone_number is not None:
        updates["phone_number"] = data.phone_number.strip()
    if data.age is not None:
        updates["age"] = data.age
    if data.gender is not None:
        updates["gender"] = data.gender
    if data.height_cm is not None:
        updates["height_cm"] = data.height_cm
    if data.weight_kg is not None:
        updates["weight_kg"] = data.weight_kg
    if data.favorite_food is not None:
        updates["favorite_food"] = data.favorite_food.strip()
    if data.new_password and len(data.new_password.strip()) >= 4:
        updates["password_hash"] = hash_password(data.new_password.strip())

    updated_user = db.update_user(ident, updates) or user

    # Sync legacy JSON
    users = load_users()
    user_key = find_user_key(users, ident) or ident
    users[user_key] = updated_user
    save_users(users)

    return {
        "status": "success",
        "message": "Profile updated successfully.",
        "database_backend": db.backend,
        "user": to_safe_user(updated_user)
    }

@app.get("/api/user/{identifier}")
def get_user_profile(identifier: str):
    user = db.get_user(identifier)
    if not user:
        users = load_users()
        user_key = find_user_key(users, identifier)
        if user_key:
            user = users[user_key]
    if not user:
        raise HTTPException(status_code=404, detail="Patient profile not found.")
    return to_safe_user(user)

@app.post("/api/upload_avatar")
async def upload_avatar(
    identifier: Optional[str] = Form(None),
    phone_number: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        ext = ".jpg"

    ident = (identifier or email or phone_number or "patient").strip()
    safe_ident = "".join(c for c in ident if c.isalnum()) or "patient"
    filename = f"{safe_ident}_{int(time.time())}{ext}"
    dest = AVATARS_DIR / filename

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    avatar_url = f"/static/avatars/{filename}"

    # Persist in DB and legacy JSON
    db.update_avatar(ident, avatar_url)
    users = load_users()
    user_key = find_user_key(users, ident)
    if user_key and user_key in users:
        users[user_key]["avatar_url"] = avatar_url
        save_users(users)

    return {
        "status": "success",
        "avatar_url": avatar_url,
        "database_backend": db.backend,
        "message": "Avatar uploaded and registered."
    }


# =============================================================================
# XGBoost Diagnostic Inference & Prescription
# =============================================================================
@app.post("/predict")
def predict_and_prescribe(data: TelemetryInput):
    global latest_telemetry

    # 1. Feature Engineering
    height_m = data.height_cm / 100.0
    bmi = round(data.weight_kg / (height_m ** 2), 2)

    features = np.array([[
        data.age,
        data.gender,
        data.height_cm,
        bmi,
        data.heart_rate,
        data.spo2,
        data.temp_c
    ]])

    # 2. XGBoost Inference
    if model is not None:
        pred_class = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()
    else:
        # Fallback heuristic if model file is missing
        if data.spo2 < 92: pred_class = 3
        elif data.temp_c > 38.3: pred_class = 1
        elif data.heart_rate > 105: pred_class = 2
        else: pred_class = 0
        probabilities = [0.25, 0.25, 0.25, 0.25]

    # 3. Pull Recommendations
    rec_entry = HEALTH_RECOMMENDATIONS.get(pred_class, HEALTH_RECOMMENDATIONS[0])

    response = {
        "status": "success",
        "predicted_condition": rec_entry["condition"],
        "class_id": pred_class,
        "description": rec_entry["description"],
        "confidence_scores": probabilities,
        "metrics": {
            "hr": data.heart_rate,
            "spo2": data.spo2,
            "temp": data.temp_c,
            "condition": rec_entry["condition"],
            "bmi": bmi,
            "max_safe_hr": 220 - data.age,
            "hr_status": "High" if data.heart_rate > 100 else ("Low" if data.heart_rate < 55 else "Normal"),
            "spo2_status": "Critical" if data.spo2 < 92 else "Optimal",
            "temp_status": "Fever" if data.temp_c > 38.0 else "Normal"
        }
    }

    mode = data.treatment_mode.lower()
    if mode == "natural":
        response["recommendations"] = {"natural_cure": rec_entry["natural_cure"]}
    elif mode == "medical":
        response["recommendations"] = {"medical_cure": rec_entry["medical_cure"]}
    else:
        response["recommendations"] = {
            "natural_cure": rec_entry["natural_cure"],
            "medical_cure": rec_entry["medical_cure"]
        }

    latest_telemetry = response

    # Persist in DB (PostgreSQL or SQLite fallback)
    try:
        db.save_telemetry(response)
    except Exception as e:
        print(f"[!] Telemetry DB save notice: {e}")

    return response

@app.get("/latest")
def get_latest_reading():
    if not latest_telemetry:
        return {"status": "waiting_for_esp32_data"}
    return latest_telemetry

# =============================================================================
# Frontend HTML Static Routing
# =============================================================================
PAGES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/triage.html": "triage.html",
    "/personal.html": "personal.html",
    "/profile.html": "profile.html",
    "/register.html": "register.html",
    "/privacy.html": "privacy.html",
    "/terms.html": "terms.html",
}

@app.get("/queries.html")
async def redirect_queries():
    return RedirectResponse("https://vasuki-2-0.vercel.app", status_code=307)

def create_page_handler(filepath: str):
    async def page_handler():
        return FileResponse(filepath)
    return page_handler

for route, filename in PAGES.items():
    app.add_api_route(route, create_page_handler(filename), methods=["GET"], response_class=FileResponse)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)