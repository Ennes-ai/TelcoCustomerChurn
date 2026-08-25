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