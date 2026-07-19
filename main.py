import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
"""
HEDEF
"Bir müşterinin özelliklerine bakarak, 
o müşterinin gelecekte şirketten ayrılıp ayrılmayacağını tahmin eden bir model kurmak."
"""

def Create_a_Model(DataFrame : pd.DataFrame):
    X = DataFrame.drop("Ayrildi_Mi", axis = 1)
    y = DataFrame["Ayrildi_Mi"]

    # OLS TABLOSU
    X_ols = sm.add_constant(X.astype(float))
    model = sm.OLS(y, X_ols).fit()
    print(model.summary())

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size= 0.2 , random_state = 42)

    scaler = StandardScaler()
    Scaled_X_Train = scaler.fit_transform(train_X) # !Traine özel fit_transform uyguluyoruz.
    Scaled_X_Test = scaler.transform(test_X)

    smote = SMOTE(random_state = 42) # 1 sınıfı azınlıktı model tam öğrenemiyordu bizde 1 sınıfı için sentetik veri üretiyoruz
    Smote_Scaled_X_Train , Smote_Train_y = smote.fit_resample(Scaled_X_Train , train_y) # sadece train için

    log_reg = LogisticRegression(max_iter= 1000 , class_weight= 'balanced')
    log_reg.fit(Smote_Scaled_X_Train, Smote_Train_y)

    prediction = log_reg.predict(Scaled_X_Test)

    katsayılar = pd.DataFrame({
        'Özellik' : X.columns,
        'Katsayi' : log_reg.coef_[0]
    }).sort_values('Katsayi',ascending=False)

    print("-" * 80)
    print(f"TAHMİN \n{prediction}")
    print("-" * 80)
    print(f"Confusion Matrix ★ \n {confusion_matrix(test_y, prediction)}")
    print("-" * 80)
    print(f"Classification Report 🪽 \n {classification_report(test_y, prediction)}")
    print("-" * 80)
    print(f"Katsayılar \n{katsayılar}")

def rename_colums(DataFrame):
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
    return DataFrame.rename(columns = column_translation)


def main(DataCSV : str):
    Data = pd.read_csv(DataCSV)
    DataFrame = pd.DataFrame(Data)
    DataFrame = rename_colums(DataFrame) # sütün isimlerini türkçeleştiriyoruz
    sorunlu = pd.to_numeric(DataFrame['Toplam_Ucret'], errors='coerce')
    DataFrame['Toplam_Ucret'] = sorunlu
    DataFrame['Toplam_Ucret'] = DataFrame['Toplam_Ucret'].fillna(0)
    DataFrame = DataFrame.drop('Musteri_ID' , axis = 1) # ID tahmin için gereksiz olduğundan dolayı atıyoruz
    Yes_NO = ['Ayrildi_Mi',
              'Esi_Var_Mi',
              'Bakmakla_Yukumlu_Kisi_Var_Mi',
              'Telefon_Hizmeti',
              'Kagitsiz_Fatura',
              ]
    cok_degerli = [
        'Birden_Fazla_Hat', 'Internet_Servis_Tipi', 'Online_Guvenlik',
        'Online_Yedekleme', 'Cihaz_Korumasi', 'Teknik_Destek',
        'TV_Yayin_Hizmeti', 'Film_Yayin_Hizmeti', 'Sozlesme_Tipi', 'Odeme_Yontemi',
    ]

    for column in Yes_NO:
        DataFrame[column] = DataFrame[column].map({'Yes' : 1 , 'No' : 0})
    DataFrame['Cinsiyet'] = DataFrame['Cinsiyet'].map({'Female' : 1 , 'Male' : 0})

    DataFrame = pd.get_dummies(data = DataFrame, columns= cok_degerli , drop_first = True)

    islenmis_sütunlar = Yes_NO + ['Cinsiyet'] + cok_degerli

    for i in DataFrame.columns:
        if i not in islenmis_sütunlar:
            print(f"Sütun {DataFrame[i]} -> {DataFrame[i].unique()}")



    #print(f"İlk 10 satır \n{DataFrame.head(n = 10)}")
    #print("-" * 80)
    #print(f"DataFrame açıklama \n {DataFrame.describe()}")
    #print("-" * 80)
    #print(f"{DataFrame.info()} \n shape {DataFrame.shape} \n {DataFrame["Toplam_Ucret"]}")
    # Sayıya çevirmeyi dene, çeviremediklerini NaN yap ve say

    #print("İşlenmemiş (hâlâ metin) sütunlar:")
    #print(DataFrame.select_dtypes(include='str').columns.tolist())
    #print(DataFrame.isnull().sum())
    tekrar_eden_sutunlar = [
        'Cihaz_Korumasi_No internet service',
        'Film_Yayin_Hizmeti_No internet service',
        'Teknik_Destek_No internet service',
        'TV_Yayin_Hizmeti_No internet service',
        'Online_Guvenlik_No internet service',
        'Online_Yedekleme_No internet service',
    ]
    DataFrame = DataFrame.drop(columns=tekrar_eden_sutunlar)
    """
    bir matrisin özdeğerlerinden biri sıfırsa (ya da sıfıra çok yakınsa), 
    o matris tekil (singular) demektir — yani matrisin tersi alınamaz (ya da almak matematiksel olarak çok kararsız 
    hale gelir). Ve bir matris Satırları/sütunları arasında lineer bağımlılık olduğunda — 
    yani bir sütun, diğerlerinin bir kombinasyonu olarak yazılabiliyorsa.
    
    Bizde ise Katsayıları yazdırdığımızda 7 sütun'un katsayısı aynı çıktı yani lineer bağımlı
     7 sütun birebir aynı (hepsi -0.116158 katsayı alıyor 
    çünkü hepsi aynı bilgiyi taşıyor). Matematiksel olarak bu, X matrisinin sütunlarından 
    6 tanesinin gereksiz olduğu anlamına geliyor
    bundan dolayı bu 6 tanesini çıkarıyoruz 
    Bunu ise şöyle fark ettik bu 7 sütunun lineer bağımlı olması OLS tablosunda
    Cond. No. ' yu arttırdı  8.37e+17 gibi yüksek sayıya bundan dolayı çıkarıyoruz.
    """
    # --- Veri Doğrulama Bloğu ---
    assert DataFrame.select_dtypes(include=['object', 'str']).columns.tolist() == [], "Encode edilmemiş sütun var!"
    assert DataFrame.isnull().sum().sum() == 0, "Eksik değer var!"
    assert len(DataFrame) == 7043, "Satır sayısı beklenenden farklı!"
    assert set(DataFrame['Ayrildi_Mi'].unique()) == {0, 1}, "Hedef sütun bozuk!"

    print("✅ Tüm veri doğrulamaları başarılı, model kurmaya hazır.")
    print(DataFrame[['Musterilik_Suresi_Ay', 'Aylik_Ucret', 'Toplam_Ucret']].corr())
    """
    Katsayılarda Toplam_Ucret değeri de çok yüksek çıktı ama Aylik_ücret ve Musterilik_suresi_ay çok düşük
    çıktı sorun olan kısım ise bu 3 değerin arasında bir ilişki olması yani mantıkken
    Toplam_Ucret = Musterilik_Suresi_Ay * Aylik_Ucret
    bundan doalyı bu üçünün arasındaki korelasyona (ilişkiye) bakıyoruz eğer
    +0.9 gibi çıkarsa toplam_ucreti atmamız gerekiyor.
    """
    DataFrame = DataFrame.drop(columns=['Toplam_Ucret'])
    """
    Attık neredeyse 1 çıktı.
    """

    Create_a_Model(DataFrame)


if __name__ == '__main__':
    main(DataCSV ="WA_Fn-UseC_-Telco-Customer-Churn.csv")