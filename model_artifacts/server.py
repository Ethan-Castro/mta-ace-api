# server.py
from fastapi import FastAPI, HTTPException
import os, json, joblib, pandas as pd

app = FastAPI(title="MTA ACE Models")
ART_DIR = os.environ.get("MODEL_ARTIFACTS_DIR", "model_artifacts")

def load_json(name):
    p = os.path.join(ART_DIR, name)
    try:
        with open(p, "r") as f: return json.load(f)
    except Exception: return None

def load_joblib(name):
    p = os.path.join(ART_DIR, name)
    try: return joblib.load(p)
    except Exception: return None

forecasts   = load_json("forecasts.json")
snapshot    = load_json("snapshot.json")
xgb         = load_joblib("xgb_risk.pkl")
xmeta       = load_json("xgb_meta.json")
hotspots_gj = load_json("hotspots.geojson")
survival    = load_json("survival.json")

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
    }

@app.get("/forecast/{route_id}")
def route_forecast(route_id: str):
    if not forecasts: raise HTTPException(503, "forecasts.json not available")
    if route_id not in forecasts: raise HTTPException(404, f"No forecast for {route_id}")
    return forecasts[route_id]

@app.get("/risk/score")
def risk_score(avg_speed_mph: float, trips_per_hour: float):
    if xgb is None: raise HTTPException(503, "xgb_risk.pkl not available")
    cols = (xmeta or {}).get("feature_cols", ["avg_speed_mph", "trips_per_hour"])
    X = pd.DataFrame([[avg_speed_mph, trips_per_hour]], columns=cols)
    try:
        score = float(xgb.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, f"Model predict failed: {e}")
    return {"risk_score": score}

@app.get("/risk/top")
def risk_top(limit: int = 100):
    p = os.path.join(ART_DIR, "top_candidates.json")
    if not os.path.exists(p): raise HTTPException(404, "No precomputed candidates")
    try: df = pd.read_json(p)
    except Exception as e: raise HTTPException(500, f"Failed to read candidates: {e}")
    return df.head(limit).to_dict(orient="records")

@app.get("/hotspots.geojson")
def hotspots_geojson():
    if hotspots_gj is None: raise HTTPException(404, "hotspots.geojson not available")
    return hotspots_gj

@app.get("/survival/km")
def survival_km():
    if survival is None or "km" not in survival: raise HTTPException(404, "KM not available")
    return survival["km"]

@app.get("/survival/cox_summary")
def survival_cox_summary():
    if survival is None or "cox_summary" not in survival: raise HTTPException(404, "Cox summary not available")
    return survival["cox_summary"]
