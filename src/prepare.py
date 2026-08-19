import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COLUMN_TRANSLATION = {  "customerID": "Musteri_ID",
            "gender": "Cinsiyet",
            "SeniorCitizen": "Yasli_Vatandas",
            "Partner": "Esi_Var_Mi",
            "Dependents": "Bakmakla_Yukumlu_Kisi_Var_Mi",
            "tenure": "Musterilik_Suresi_Ay",
            "PhoneService": "Telefon_Hizmeti",
            "MultipleLines": "Birden_Fazla_Hat",
            "InternetService": "Internet_Servis_Tipi",
            "OnlineSecurity": "Online_Guvenlik",
            "OnlineBackup": "Online_Yedekleme",
            "DeviceProtection": "Cihaz_Korumasi",
            "TechSupport": "Teknik_Destek",
            "StreamingTV": "TV_Yayin_Hizmeti",
            "StreamingMovies": "Film_Yayin_Hizmeti",
            "Contract": "Sozlesme_Tipi",
            "PaperlessBilling": "Kagitsiz_Fatura",
            "PaymentMethod": "Odeme_Yontemi",
            "MonthlyCharges": "Aylik_Ucret",
            "TotalCharges": "Toplam_Ucret",
            "Churn": "Ayrildi_Mi",
            }  # HEDEF SÜTUN (y)   # mevcut sözlüğün, aynen kalsın

# Adım 3'ten sonra hepsi Yes/No oldu
BINARY_COLS = [
    'Esi_Var_Mi', 'Bakmakla_Yukumlu_Kisi_Var_Mi', 'Telefon_Hizmeti', 'Kagitsiz_Fatura',
    'Birden_Fazla_Hat', 'Online_Guvenlik', 'Online_Yedekleme', 'Cihaz_Korumasi',
    'Teknik_Destek', 'TV_Yayin_Hizmeti', 'Film_Yayin_Hizmeti',
]

CATEGORICAL_COLS = ['Internet_Servis_Tipi', 'Sozlesme_Tipi', 'Odeme_Yontemi']

NUMERIC_COLS = ['Musterilik_Suresi_Ay', 'Aylik_Ucret', 'Yasli_Vatandas', 'Cinsiyet'] + BINARY_COLS


class RowCleaner(BaseEstimator, TransformerMixin):
    """
    Sadece satır-içi (Tip 1) temizlik yapar.
    Veriden hiçbir şey öğrenmediği için fit'in boş olması BURADA doğrudur.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.rename(columns=COLUMN_TRANSLATION).copy()

        # ID tahmin için işe yaramaz; Toplam_Ucret multicollinearity yüzünden atılıyor
        df = df.drop(columns=[c for c in ('Musteri_ID', 'Toplam_Ucret') if c in df.columns])

        # "No internet service" / "No phone service" -> "No"
        df[BINARY_COLS] = df[BINARY_COLS].replace(
            {'No internet service': 'No', 'No phone service': 'No'}
        )
        df[BINARY_COLS] = df[BINARY_COLS].apply(lambda s: s.map({'Yes': 1, 'No': 0}))
        df['Cinsiyet'] = df['Cinsiyet'].map({'Female': 1, 'Male': 0})

        return df


def validate_raw_data(df):
    assert set(df['Churn'].unique()) == {'Yes', 'No'}, "Hedef sütun bozuk!"
