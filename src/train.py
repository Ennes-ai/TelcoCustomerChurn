import json
from datetime import datetime
from pathlib import Path


import joblib
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare import RowCleaner, NUMERIC_COLS, CATEGORICAL_COLS, validate_raw_data

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "model"

def build_pipeline():
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False),CATEGORICAL_COLS)
    ],remainder="drop")
    
    return Pipeline([
        ("cleaner", RowCleaner()),
        ("prep",    preprocessor),
        ("smote",   SMOTE(random_state=42)),
        ("log_reg", LogisticRegression(max_iter=1000)),
    ])


def main():
    raw = pd.read_csv(DATA_PATH)
    validate_raw_data(raw)

    X = raw.drop("Churn", axis=1)
    y = raw["Churn"].map({"Yes": 1, "No": 0})

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(train_X, train_y)

    rapor = classification_report(test_y, pipe.predict(test_X), output_dict=True)
    print(classification_report(test_y, pipe.predict(test_X)))

    # kabul testi: modelin görmediği tek satır
    assert pipe.predict_proba(test_X.iloc[[0]]).shape == (1, 2)
    
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "pipeline.joblib")
    
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "egitim_tarihi": datetime.now().isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "n_train": len(train_X),
        "recall_1": round(rapor["1"]["recall"], 3),
        "precision_1": round(rapor["1"]["precision"], 3),
        "girdi_sutunlari": list(X.columns),
    },indent=2, ensure_ascii=False),encoding="utf-8")
    
    print("✅ model/pipeline.joblib yazıldı")

if __name__ == "__main__":
    main()