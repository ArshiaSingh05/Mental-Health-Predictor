from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import webbrowser, threading, os
from pathlib import Path

from model_utils import preprocess, model


app = FastAPI(title="Mental‑Health Predictor API")

# -------------------------------------------------
# 1)  Serve the built Vite frontend  (…/cognifit/frontend/dist)
# -------------------------------------------------
frontend_dist = (Path(__file__).resolve().parent.parent      # …/backend
                 / "cognifit" / "frontend" / "dist")
if not frontend_dist.exists():
    raise RuntimeError("⚠️  Build the frontend first: cd cognifit/frontend && npm run build")

app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")


# -------------------------------------------------
# 2)  Input schema
# -------------------------------------------------
class InputData(BaseModel):
    age: int
    no_employees_mid: int
    leave: int
    work_interfere: int
    self_employed: int
    family_history: int
    remote_work: int
    tech_company: int
    benefits: int
    care_options: int
    wellness_program: int
    seek_help: int
    anonymity: int
    mental_health_interview: int
    phys_health_interview: int
    mental_vs_physical: int
    mental_health_consequence: int
    phys_health_consequence: int
    obs_consequence: int
    coworkers: int
    supervisor: int
    gender: str                      # "male"|"female"|"other"


# -------------------------------------------------
# 3)  Prediction route
# -------------------------------------------------
@app.post("/predict/")
def predict(data: InputData):
    try:
        df = preprocess(data.dict())
        pred = model.predict(df)[0]
        risk = "High risk" if pred == 1 else "Low risk"
        return {"prediction": risk}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# 4)  Auto‑open browser once server starts
# -------------------------------------------------
def launch_browser():
    webbrowser.open("http://127.0.0.1:8000")
threading.Timer(1.5, launch_browser).start()
