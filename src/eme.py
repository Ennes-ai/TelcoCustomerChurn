import pandas as pd
from prepare import RowCleaner
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    

raw = pd.read_csv(DATA_PATH)
X = raw.drop("Churn", axis=1)

cleaner = RowCleaner()

print(cleaner.transform(X).shape)            # tüm veri
print(cleaner.transform(X.iloc[:1]).shape)   # tek müşteri


from prepare import RowCleaner
c = RowCleaner()
print(c.transform(X).shape)          # (7043, 18) civarı
print(c.transform(X.iloc[:1]).shape) # (1, 18) — AYNI olmalı