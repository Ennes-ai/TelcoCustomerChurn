import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException

from schema import Musteri


MODEL_DIR = Path(__file__).resolve().parent / "model"

pipe = joblib.load(MODEL_DIR / "pipeline.joblib")
meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))



if meta["sklearn"] != sklearn.__version__:
    raise RuntimeError(
         f"Model {meta['sklearn']} ile eğitildi, ortamda {sklearn.__version__} var."
    )
    
VARSAYILAN_ESIK = 0.20

app = FastAPI(
    title="Telco Churn API",
    description="Müşteri özelliklerinden ayrılma olasılığı tahmin eder.",
    version="1.0",
)


@app.get("/health")
def health():
    return {"durum": "ayakta", "varsayilan_esik": VARSAYILAN_ESIK, **meta}

@app.post("/predict")
def predict(musteri : Musteri, esik:float = VARSAYILAN_ESIK):
    df = pd.DataFrame([musteri.model_dump()])
    
    try:
        olasilik = float(pipe.predict_proba(df)[0 , 1])
        
    except ValueError as e:
        raise HTTPException(status_code=422 , detail=str(e))
    
    
    return{
        "ayrilma_olasiligi": round(olasilik, 4),
        "esik": esik,
        "karar": "ayrilir" if olasilik >= esik else "kalir",
    }
    