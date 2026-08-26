import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from slowapi import Limiter ,_rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .schema import Musteri
from fastapi import Request


MODEL_DIR = Path(__file__).resolve().parent / "Model"

pipe = joblib.load(MODEL_DIR / "pipeline.joblib")
meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))

limiter = Limiter(key_func=get_remote_address)


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
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/health")
def health():
    return {"durum": "ayakta", "varsayilan_esik": VARSAYILAN_ESIK, **meta}

@app.post("/predict")
@limiter.limit("30/minute")
def predict(request :Request ,musteri : Musteri, esik:float = VARSAYILAN_ESIK):
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
    