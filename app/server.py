"""
app/server.py
──────────────
FastAPI backend — serves the index.html frontend and
exposes /predict endpoint that calls predict_model.py.

Run:
    pip install fastapi uvicorn
    uvicorn app.server:app --reload --port 8000
Then open: http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models.predict_model import predict

app = FastAPI(title="LoanIQ API", version="1.0.0")


class ApplicantInput(BaseModel):
    ApplicantIncome:   float
    CoapplicantIncome: float
    LoanAmount:        float
    Loan_Amount_Term:  float
    Credit_History:    float
    Gender:            str
    Married:           str
    Dependents:        str
    Education:         str
    Self_Employed:     str
    Property_Area:     str


@app.get("/")
def serve_app():
    return FileResponse("app/index.html")


@app.post("/predict")
def predict_endpoint(data: ApplicantInput):
    raw = data.dict()
    result = predict(raw)
    return JSONResponse(result)


@app.get("/health")
def health():
    return {"status": "ok", "model": "Random Forest", "threshold": 0.38}


# Mount static files (css, favicon)
if os.path.exists("app/static"):
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
