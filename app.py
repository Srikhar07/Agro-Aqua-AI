"""
AgroAqua AI — Complete Flask Backend with SQLite
=================================================
Place this file alongside:
    templates/      static/      best_farm_models/      best_aqua_models/

Install:
    pip install flask joblib scikit-learn numpy qrcode[pil] pillow

Run:
    python app.py   →   open http://127.0.0.1:5000
"""

import os
import io
import base64
import datetime
import sqlite3
import warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, g
)

try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# =============================================================================
# Flask app
# =============================================================================
app = Flask(__name__)
app.secret_key = "agroaqua_super_secret_2024"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FARM_DIR = os.path.join(BASE_DIR, "best_farm_models")
AQUA_DIR = os.path.join(BASE_DIR, "best_aqua_models")
DB_PATH  = os.path.join(BASE_DIR, "agroaqua.db")

# =============================================================================
# SQLite helpers
# =============================================================================

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db

@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            email      TEXT    NOT NULL UNIQUE,
            phone      TEXT    NOT NULL,
            age        INTEGER NOT NULL,
            place      TEXT    NOT NULL,
            password   TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            module     TEXT    NOT NULL,
            inputs     TEXT,
            result     TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reports (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            module     TEXT    NOT NULL,
            result     TEXT    NOT NULL,
            qr_data    TEXT,
            created_at TEXT    NOT NULL
        );
        """)
        db.commit()

# =============================================================================
# Model loader (lazy cache)
# =============================================================================
_cache = {}

def _load(path):
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]

def farm(name): return _load(os.path.join(FARM_DIR, name))
def aqua(name): return _load(os.path.join(AQUA_DIR, name))

def run_model(model_obj, scaler_obj, feature_list):
    X  = np.array(feature_list, dtype=float).reshape(1, -1)
    Xs = scaler_obj.transform(X)
    return model_obj.predict(Xs)[0]

# =============================================================================
# QR helper
# =============================================================================

def make_qr(text):
    if not QR_AVAILABLE or not text:
        return ""
    img = qrcode.make(str(text))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# =============================================================================
# Auth / DB helpers
# =============================================================================

def logged_in():
    return "user_id" in session

def require_login():
    if not logged_in():
        flash("Please log in first.")
        return redirect(url_for("login"))
    return None

def current_user():
    if not logged_in():
        return None
    return get_db().execute("SELECT * FROM users WHERE id=?",
                            (session["user_id"],)).fetchone()

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

def save_history(module, inputs_str, result_str):
    if not logged_in():
        return
    db = get_db()
    db.execute(
        "INSERT INTO history (user_id,module,inputs,result,created_at) VALUES (?,?,?,?,?)",
        (session["user_id"], module, inputs_str, result_str, now_str())
    )
    db.commit()

def save_report(module, result_str):
    if not logged_in():
        return
    qr = make_qr(f"AgroAqua | {module} | {result_str} | {now_str()}")
    db = get_db()
    db.execute(
        "INSERT INTO reports (user_id,module,result,qr_data,created_at) VALUES (?,?,?,?,?)",
        (session["user_id"], module, result_str, qr, now_str())
    )
    db.commit()

def get_history(limit=None):
    sql = "SELECT * FROM history WHERE user_id=? ORDER BY id DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return get_db().execute(sql, (session["user_id"],)).fetchall()

def get_reports():
    return get_db().execute(
        "SELECT * FROM reports WHERE user_id=? ORDER BY id DESC",
        (session["user_id"],)
    ).fetchall()

def count_history():
    row = get_db().execute(
        "SELECT COUNT(*) FROM history WHERE user_id=?", (session["user_id"],)
    ).fetchone()
    return row[0] if row else 0

def count_reports():
    row = get_db().execute(
        "SELECT COUNT(*) FROM reports WHERE user_id=?", (session["user_id"],)
    ).fetchone()
    return row[0] if row else 0

# =============================================================================
# Label helpers — give human-readable output
# =============================================================================

def fish_life_label(val):
    if str(val).strip() in ("1", "1.0"): return "Fish will SURVIVE"
    if str(val).strip() in ("0", "0.0"): return "Fish will NOT Survive"
    return f"Status: {val}"

def water_quality_label(val):
    m = {"0": "Poor Quality", "1": "Good Quality", "2": "Excellent Quality"}
    return m.get(str(int(float(val))), f"Grade {val}")

def potable_label(val):
    if str(val).strip() in ("1","1.0"):
        return "Potable - Safe for Crops"
    return "Not Potable - Unsafe for Crops"

def disease_label(val):
    if str(val).strip() in ("1","1.0"):
        return "Disease Present"
    return "No Disease Detected"

# =============================================================================
# Routes — Public
# =============================================================================

@app.route("/")
def home():
    return render_template("web_mainpage.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute(
            "SELECT * FROM users WHERE email=? AND password=?", (email, password)
        ).fetchone()
        if user:
            session.clear()
            session["user_id"]   = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.")
    return render_template("login_page.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name    = request.form.get("name", "").strip()
        email   = request.form.get("email", "").strip().lower()
        phone   = request.form.get("phone", "").strip()
        age     = request.form.get("age", "").strip()
        place   = request.form.get("place", "").strip()
        pw      = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not all([name, email, phone, age, place, pw]):
            flash("All fields are required.")
            return render_template("signup_page.html")
        if pw != confirm:
            flash("Passwords do not match.")
            return render_template("signup_page.html")

        db = get_db()
        if db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
            flash("Email already registered. Please log in.")
            return render_template("signup_page.html")

        db.execute(
            "INSERT INTO users (name,email,phone,age,place,password,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (name, email, phone, int(age), place, pw,
             datetime.datetime.now().strftime("%Y-%m-%d"))
        )
        db.commit()
        flash("Account created! Please log in.")
        return redirect(url_for("login"))

    return render_template("signup_page.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# =============================================================================
# Routes — Dashboard & sidebar pages
# =============================================================================

@app.route("/dashboard")
def dashboard():
    redir = require_login()
    if redir: return redir
    return render_template(
        "user_dashboard.html",
        username          = session.get("user_name", "User"),
        total_predictions = count_history(),
        reports_generated = count_reports(),
        recent_predictions= get_history(limit=5)
    )


@app.route("/profile")
def profile():
    redir = require_login()
    if redir: return redir
    user = current_user()
    qr   = make_qr(
        f"AgroAqua User | {user['name']} | {user['email']} | {user['phone']}"
    )
    return render_template("profile.html", user=user, qr_dataurl=qr)


@app.route("/history")
def history():
    redir = require_login()
    if redir: return redir
    return render_template("history.html", history_data=get_history())


@app.route("/reports")
def reports():
    redir = require_login()
    if redir: return redir
    return render_template("reports.html", reports_data=get_reports())


@app.route("/scan-qr")
def scan_qr():
    redir = require_login()
    if redir: return redir
    user = current_user()
    qr   = make_qr(
        f"AgroAqua User | {user['name']} | {user['email']} | {user['phone']}"
    )
    return render_template("scan_qr.html", qr_dataurl=qr, user=user)

# =============================================================================
# Routes — Agriculture Modules
# =============================================================================

@app.route("/crop-recommendation", methods=["GET", "POST"])
def crop_recommendation():
    redir = require_login()
    if redir: return redir
    prediction = None
    if request.method == "POST":
        try:
            f = [
                float(request.form["nitrogen"]),
                float(request.form["phosphorus"]),
                float(request.form["potassium"]),
                float(request.form["temperature"]),
                float(request.form["humidity"]),
                float(request.form["ph"]),
                float(request.form["rainfall"]),
            ]
            raw = run_model(farm("crop_recommendation.joblib"),
                            farm("crop_recommendation_scaler.joblib"), f)
            prediction = f"Recommended Crop: {str(raw).title()}"
            inp = (f"N={f[0]}, P={f[1]}, K={f[2]}, Temp={f[3]}, "
                   f"Humidity={f[4]}, pH={f[5]}, Rainfall={f[6]} mm")
            save_history("Crop Recommendation", inp, prediction)
            save_report("Crop Recommendation", prediction)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("crop_recommendation.html", prediction=prediction)


@app.route("/crop-yield", methods=["GET", "POST"])
def crop_yield():
    redir = require_login()
    if redir: return redir
    prediction = None
    if request.method == "POST":
        try:
            f = [
                float(request.form["rain_fall"]),
                float(request.form["fertilizer"]),
                float(request.form["temperatue"]),
                float(request.form["nitrogen"]),
                float(request.form["phosphorus"]),
                float(request.form["potassium"]),
            ]
            raw = run_model(farm("crop_yield_prediction.joblib"),
                            farm("crop_yield_scaler.joblib"), f)
            prediction = f"Predicted Yield: {float(raw):.2f} Quintals/Acre"
            inp = (f"Rainfall={f[0]} mm, Fertilizer={f[1]}, Temp={f[2]}°C, "
                   f"N={f[3]}, P={f[4]}, K={f[5]}")
            save_history("Crop Yield Prediction", inp, prediction)
            save_report("Crop Yield Prediction", prediction)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("crop_yield.html", prediction=prediction)


@app.route("/crop-water-test", methods=["GET", "POST"])
def crop_water_test():
    redir = require_login()
    if redir: return redir
    prediction = None
    if request.method == "POST":
        try:
            f = [
                float(request.form["ph"]),
                float(request.form["hardness"]),
                float(request.form["solids"]),
                float(request.form["chloramines"]),
                float(request.form["sulfate"]),
                float(request.form["organic_carbon"]),
                float(request.form["trihalomethanes"]),
                float(request.form["turbidity"]),
            ]
            raw = run_model(farm("crop_water_test.joblib"),
                            farm("crop_water_scaler.joblib"), f)
            prediction = potable_label(raw)
            inp = (f"pH={f[0]}, Hardness={f[1]}, Solids={f[2]}, "
                   f"Chloramines={f[3]}, Sulfate={f[4]}, "
                   f"OrganicCarbon={f[5]}, THM={f[6]}, Turbidity={f[7]}")
            save_history("Crop Water Test", inp, prediction)
            save_report("Crop Water Test", prediction)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("crop_water_test.html", prediction=prediction)


@app.route("/plant-disease", methods=["GET", "POST"])
def plant_disease():
    redir = require_login()
    if redir: return redir
    prediction = None
    if request.method == "POST":
        try:
            f = [
                float(request.form["temperature"]),
                float(request.form["humidity"]),
                float(request.form["rainfall"]),
                float(request.form["soil_pH"]),
            ]
            raw = run_model(farm("plant_disease_model.joblib"),
                            farm("plant_disease_scaler.joblib"), f)
            prediction = disease_label(raw)
            inp = (f"Temp={f[0]}°C, Humidity={f[1]}%, "
                   f"Rainfall={f[2]} mm, Soil_pH={f[3]}")
            save_history("Plant Disease Detection", inp, prediction)
            save_report("Plant Disease Detection", prediction)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("plant_disease.html", prediction=prediction)

# =============================================================================
# Routes — Aquaculture Modules
# =============================================================================

@app.route("/fish-life", methods=["GET", "POST"])
def fish_life():
    redir = require_login()
    if redir: return redir
    prediction = None
    if request.method == "POST":
        try:
            f = [
                float(request.form["nitrate"]),
                float(request.form["ph"]),
                float(request.form["ammonia"]),
                float(request.form["temp"]),
                float(request.form["do"]),
                float(request.form["turbidity"]),
                float(request.form["manganese"]),
                float(request.form["pressure"]),
                float(request.form["tempc"]),
                float(request.form["humidity"]),
                float(request.form["windspeed"]),
            ]
            raw = run_model(aqua("fish_life_model.joblib"),
                            aqua("fish_life_scaler.joblib"), f)
            prediction = fish_life_label(raw)
            inp = (f"Nitrate={f[0]} PPM, pH={f[1]}, Ammonia={f[2]} mg/L, "
                   f"Temp={f[3]}, DO={f[4]}, Turbidity={f[5]}, "
                   f"Manganese={f[6]} mg/L, Pressure={f[7]}, TempC={f[8]}, "
                   f"Humidity={f[9]}%, Wind={f[10]} Kmph")
            save_history("Fish Life Prediction", inp, prediction)
            save_report("Fish Life Prediction", prediction)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("fish_life.html", prediction=prediction)


@app.route("/water-quality", methods=["GET", "POST"])
def water_quality():
    redir = require_login()
    if redir: return redir
    prediction = None
    if request.method == "POST":
        try:
            f = [
                float(request.form["temp"]),
                float(request.form["turbidity"]),
                float(request.form["do_mg_l"]),
                float(request.form["bod"]),
                float(request.form["co2"]),
                float(request.form["ph"]),
                float(request.form["alkalinity"]),
                float(request.form["hardness"]),
                float(request.form["calcium"]),
                float(request.form["ammonia"]),
                float(request.form["nitrite"]),
                float(request.form["phosphorus"]),
                float(request.form["h2s"]),
                float(request.form["plankton"]),
            ]
            raw = run_model(aqua("WQD.joblib"),
                            aqua("WQD_scaler.joblib"), f)
            prediction = f"Water Quality: {water_quality_label(raw)}"
            inp = (f"Temp={f[0]}, Turbidity={f[1]} cm, DO={f[2]} mg/L, "
                   f"BOD={f[3]} mg/L, CO2={f[4]}, pH={f[5]}, "
                   f"Alkalinity={f[6]}, Hardness={f[7]}, Calcium={f[8]}, "
                   f"Ammonia={f[9]}, Nitrite={f[10]}, Phosphorus={f[11]}, "
                   f"H2S={f[12]}, Plankton={f[13]}")
            save_history("Water Quality Prediction", inp, prediction)
            save_report("Water Quality Prediction", prediction)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("water_quality.html", prediction=prediction)

# =============================================================================
# Context processor
# =============================================================================

@app.context_processor
def inject_globals():
    return dict(now=datetime.datetime.now)

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    init_db()
    print("\n" + "="*58)
    print("  AgroAqua AI — Starting up")
    print(f"  Database : {DB_PATH}")
    print("  URL      : http://127.0.0.1:5000")
    print("="*58)

    checks = [
        (FARM_DIR, "crop_recommendation.joblib"),
        (FARM_DIR, "crop_recommendation_scaler.joblib"),
        (FARM_DIR, "crop_yield_prediction.joblib"),
        (FARM_DIR, "crop_yield_scaler.joblib"),
        (FARM_DIR, "crop_water_test.joblib"),
        (FARM_DIR, "crop_water_scaler.joblib"),
        (FARM_DIR, "plant_disease_model.joblib"),
        (FARM_DIR, "plant_disease_scaler.joblib"),
        (AQUA_DIR, "fish_life_model.joblib"),
        (AQUA_DIR, "fish_life_scaler.joblib"),
        (AQUA_DIR, "WQD.joblib"),
        (AQUA_DIR, "WQD_scaler.joblib"),
    ]
    all_ok = True
    for folder, fname in checks:
        exists = os.path.exists(os.path.join(folder, fname))
        status = "OK " if exists else "MISSING"
        print(f"  [{status}] {fname}")
        if not exists:
            all_ok = False

    print()
    if all_ok:
        print("  All models ready!\n")
    else:
        print("  WARNING: Missing model files — those modules will error.\n")

    app.run(debug=True, host="0.0.0.0", port=5000)