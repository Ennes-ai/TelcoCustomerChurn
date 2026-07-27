import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Prepare(BaseEstimator, TransformerMixin): # bu kütüphaleri kullanmamız gerekiyor Scikit-Learn standartlarında bir dönüştürücünün yazmamız için ve tanımlı olması için
    def __init__(self):                         # 3 tane fonksiyon gerekli fit transform ve fit_trasform fit_transformu ise bize TransformerMixin sağlıyor
        self.Yes_NO = [                         # BaseEstimator, sınıfımıza Scikit-Learn'ün parametre yönetim araçlarını kazandırır (get_params ve set_params).
        'Ayrildi_Mi',
        'Bakmakla_Yukumlu_Kisi_Var_Mi',
        'Telefon_Hizmeti',
        'Esi_Var_Mi',
        'Kagitsiz_Fatura',
        ]
        self.cok_degerli = [
        'Birden_Fazla_Hat', 'Internet_Servis_Tipi', 'Online_Guvenlik',
        'Online_Yedekleme', 'Cihaz_Korumasi', 'Teknik_Destek',
        'TV_Yayin_Hizmeti', 'Film_Yayin_Hizmeti', 'Sozlesme_Tipi', 'Odeme_Yontemi',
        ]
        self.tekrar_eden_sutunlar = [
        'Cihaz_Korumasi_No internet service',
        'Film_Yayin_Hizmeti_No internet service',
        'Teknik_Destek_No internet service',
        'TV_Yayin_Hizmeti_No internet service',
        'Online_Guvenlik_No internet service',
        'Online_Yedekleme_No internet service',
        ]
    def fit(self , X , y = None):
        return self

    def rename_columns(self ,DataFrame: pd.DataFrame):
        column_translation = {
            "customerID": "Musteri_ID",
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
            "Churn": "Ayrildi_Mi",  # HEDEF SÜTUN (y)
        }
        return DataFrame.rename(columns=column_translation)

    def transform(self , X):
        df = X.copy()

        df = self.rename_columns(DataFrame= df)
        if 'Toplam_Ucret' in df.columns:
            df['Toplam_Ucret'] = pd.to_numeric(df['Toplam_Ucret'], errors='coerce').fillna(0)

        if 'Musteri_ID' in df.columns:
            df = df.drop('Musteri_ID', axis=1)  # ID tahmin için gereksiz olduğundan dolayı atıyoruz

        for column in self.Yes_NO:
            if column in df.columns:
                df[column] = df[column].map({'Yes': 1, 'No': 0})
        if 'Cinsiyet' in df.columns:
            df['Cinsiyet'] = df['Cinsiyet'].map({'Female': 1, 'Male': 0})

        existing_cok_degerli = [col for col in self.cok_degerli if col in df.columns]
        if existing_cok_degerli:
            df = pd.get_dummies(data=df, columns=existing_cok_degerli, drop_first=True)

        colm_to_drop = [col for col in  self.tekrar_eden_sutunlar if col in df.columns]

        if 'Toplam_Ucret' in df.columns:
            colm_to_drop.append('Toplam_Ucret')

        if colm_to_drop:
            df = df.drop(columns=colm_to_drop)

        assert df.select_dtypes(include=['object', 'str']).columns.tolist() == [], "Encode edilmemiş sütun var!"
        assert df.isnull().sum().sum() == 0, "Eksik değer var!"


        return df
def validate_raw_data(df):
    assert len(df) == 7043, "Satır sayısı beklenenden farklı!"
    assert set(df['Churn'].unique()) == {'Yes', 'No'}, "Hedef sütun bozuk!"