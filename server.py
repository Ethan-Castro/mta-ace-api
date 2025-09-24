# server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os, json, joblib, pandas as pd

# --- App & CORS ---------------------------------------------------------------
app = FastAPI(title="MTA ACE Models")

# Allow all during setup; tighten later via ALLOW_ORIGINS env (comma-separated)
_allow_env = os.environ.get("ALLOW_ORIGINS", "").strip()
ALLOW_ORIGINS: List[str] = (
    [o.strip() for o in _allow_env.split(",") if o.strip()]
    if _allow_env else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],   # only GET in this service, but * is fine
    allow_headers=["*"],
)

# --- Artifacts ---------------------------------------------------------------
ART_DIR = os.environ.get("MODEL_ARTIFACTS_DIR", "model_artifacts")

def _path(name: str) -> str:
    return os.path.join(ART_DIR, name)

def load_json(name: str) -> Any | None:
    try:
        with open(_path(name), "r") as f:
            return json.load(f)
    except Exception:
        return None

def load_joblib(name: str) -> Any | None:
    try:
        return joblib.load(_path(name))
    except Exception:
        return None

# Load once at startup (service process lifetime)
forecasts   = load_json("forecasts.json")          # { route_id: {history, forecast, ...}, ... }
snapshot    = load_json("snapshot.json")           # { snapshot_as_of: ... }
xgb         = load_joblib("xgb_risk.pkl")          # trained regressor
xmeta       = load_json("xgb_meta.json")           # { feature_cols: [...] }
hotspots_gj = load_json("hotspots.geojson")        # GeoJSON FeatureCollection
survival    = load_json("survival.json")           # { km: {...}, cox_summary: {...} }
# Optional: precomputed table
_top_candidates_path = _path("top_candidates.json")

# --- Helpers -----------------------------------------------------------------
def _ensure(cond: bool, status: int, msg: str):
    if not cond:
        raise HTTPException(status, msg)

# --- Endpoints ---------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "snapshot": snapshot or {},
        "artifacts": {
            "forecasts": forecasts is not None,
            "xgb": xgb is not None,
            "xgb_meta": xmeta is not None,
            "hotspots": hotspots_gj is not None,
            "survival": survival is not None,
        },
        "cors": {"allow_origins": ALLOW_ORIGINS},
    }

@app.get("/routes")
def routes():
    """List route IDs available in forecasts.json to help UIs populate dropdowns."""
    rts = sorted(list(forecasts.keys())) if isinstance(forecasts, dict) else []
    return {"routes": rts, "count": len(rts)}

@app.get("/forecast/{route_id}")
def route_forecast(route_id: str):
    _ensure(isinstance(forecasts, dict), 503, "forecasts.json not available")
    _ensure(route_id in forecasts, 404, f"No forecast for {route_id}")
    # payload already normalized by your exporter
    return forecasts[route_id]

@app.get("/risk/score")
def risk_score(avg_speed_mph: float, trips_per_hour: float):
    _ensure(xgb is not None, 503, "xgb_risk.pkl not available")
    cols = (xmeta or {}).get("feature_cols", ["avg_speed_mph", "trips_per_hour"])
    X = pd.DataFrame([[avg_speed_mph, trips_per_hour]], columns=cols)
    try:
        score = float(xgb.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, f"Model predict failed: {e}")
    return {"risk_score": score}

@app.get("/risk/top")
def risk_top(limit: int = 100):
    _ensure(os.path.exists(_top_candidates_path), 404, "No precomputed candidates")
    try:
        df = pd.read_json(_top_candidates_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to read candidates: {e}")
    # guard against silly inputs
    limit = max(1, min(int(limit), 200))
    return df.head(limit).to_dict(orient="records")

@app.get("/hotspots.geojson")
def hotspots_geojson():
    _ensure(hotspots_gj is not None, 404, "hotspots.geojson not available")
    return hotspots_gj

@app.get("/survival/km")
def survival_km():
    _ensure(isinstance(survival, dict) and "km" in survival, 404, "KM not available")
    return survival["km"]

@app.get("/survival/cox_summary")
def survival_cox_summary():
    _ensure(isinstance(survival, dict) and "cox_summary" in survival, 404, "Cox summary not available")
    return survival["cox_summary"]
