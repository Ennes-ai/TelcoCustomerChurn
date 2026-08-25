import pandas as pd
from prepare import RowCleaner
from pathlib import Path
from sklearn.preprocessing import StandardScaler , OneHotEncoder
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    

raw = pd.read_csv(DATA_PATH)
X = raw.drop("Churn", axis=1)


enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
enc.fit(pd.DataFrame({"x": ["a", "b", "c"]}))

print(enc.transform(pd.DataFrame({"x": ["a"]})))    # baz kategori
print(enc.transform(pd.DataFrame({"x": ["z"]})))    # hiç görülmemiş