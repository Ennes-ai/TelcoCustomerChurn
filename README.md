# Telco Customer Churn — Uçtan Uca ML Servisi

Bir telekom şirketinin müşteri verilerinden ayrılma (churn) riskini tahmin eden
sınıflandırma modeli. Notebook'ta kalmadı: eğitilmiş model bir REST API olarak
paketlendi, Docker ile konteynerize edildi ve canlıya alındı.

**Canlı servis:** https://telcocustomerchurn-87zz.onrender.com/docs

> Ücretsiz sunucu 15 dakika trafik almazsa uykuya geçer. İlk istek 50 saniye kadar
> sürebilir, sonrakiler normal hızda yanıt verir.

---

## Problem

Yeni müşteri kazanmak, mevcut müşteriyi elde tutmaktan çok daha pahalı. Ayrılma
riski taşıyan müşteriyi önceden tespit edebilen bir şirket, ona özel kampanya
sunarak kaybı önleyebilir.

Bu projede amaç yüksek accuracy değil, **kampanya bütçesini doğru kişilere
yönlendirmek**. Bu ayrım modelin her kararını şekillendirdi.

**Veri:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7043 müşteri, 21 özellik.

---

## Mimari

```
Ham JSON
   │
   ▼
Pydantic şeması        →  tip ve kategori doğrulaması, hatalıysa 422
   │
   ▼
RowCleaner             →  iş kuralları (TotalCharges boşluk düzeltmesi vb.)
   │
   ▼
ColumnTransformer      →  OneHotEncoder(handle_unknown="ignore") + StandardScaler
   │
   ▼
SMOTE                  →  yalnızca eğitimde, imblearn Pipeline içinde
   │
   ▼
LogisticRegression     →  olasılık
   │
   ▼
Eşik (0.20)            →  karar
```

Tüm zincir tek bir `pipeline.joblib` artifact'i olarak serileştirildi. API bu
dosyayı yükleyip doğrudan `predict_proba` çağırıyor — servis tarafında hiçbir
manuel dönüşüm adımı yok.

---

## Kilit Kararlar

### Neden `ColumnTransformer`, neden `get_dummies` değil

`pd.get_dummies` durumsuzdur: gördüğü veriye göre sütun üretir. Eğitim setinde 21
sütun üretirken, tek satırlık bir tahmin isteğinde 13 sütun üretir ve model
patlar. `OneHotEncoder` ise `fit` sırasında kategorileri hafızasında tutar ve her
`transform` çağrısında aynı şekli üretir.

`handle_unknown="ignore"` eğitimde görülmemiş bir kategori geldiğinde hata yerine
sıfır vektör üretir — üretimde beklenmedik girdi kaçınılmazdır.

`drop="first"` bilinçli olarak **kullanılmadı**: `handle_unknown="ignore"` ile
birlikte kullanıldığında bilinmeyen kategori ile referans kategori aynı gösterime
düşer, yani model sessizce yanlış tahmin üretir. Test edilerek doğrulandı.

### Eşik neden 0.20

Varsayılan 0.5 eşiği, iki hata tipinin eşit maliyetli olduğunu varsayar. Burada
değiller:

- **False Negative** (ayrılacak müşteriyi kaçırmak): ~500 birim, müşteri geri
  gelmiyor
- **False Positive** (kalacak müşteriye kampanya göndermek): ~50 birim, sadece
  kampanya maliyeti

Bu maliyet fonksiyonuyla 0.10–0.25 aralığı düz çıktı; ortasından 0.20 seçildi.
Sonuç: modelli strateji, "herkese kampanya gönder" temel stratejisine göre
toplam maliyeti **%30 azaltıyor**.

Eşik test setinden seçildi; bağımsız bir doğrulama seti kullanılmadı. Gerçek bir
projede eşik ayrı bir validation seti üzerinde belirlenmelidir.

### Dengesiz veri

Müşterilerin %73'ü kalıyor, %27'si ayrılıyor. "Herkes kalacak" diyen boş bir model
bile %73 accuracy alır — bu yüzden accuracy tek başına yanıltıcı, Recall ve
Precision'a odaklanıldı.

| Yaklaşım | FN | FP | Recall (1) | Precision (1) | Accuracy |
|---|---|---|---|---|---|
| Varsayılan | 150 | 103 | 0.60 | 0.68 | 0.82 |
| `class_weight='balanced'` | 66 | 290 | 0.82 | 0.51 | 0.75 |
| SMOTE | 62 | 284 | 0.83 | 0.52 | 0.75 |

SMOTE yalnızca train setine, imblearn Pipeline içinde uygulandı. Bölme öncesi
uygulansaydı sentetik örnekler test setine sızar ve sonuçlar yanıltıcı derecede
iyi görünürdü.

### Multicollinearity

`statsmodels` OLS tanı tablosunda `Cond. No.` 8.37e+17 çıktı — tasarım matrisinde
güçlü doğrusal bağımlılık işareti. İki kaynak bulundu:

1. Yedi farklı sütundaki `"No internet service"` kategorisi birebir aynı bilgiyi
   taşıyordu. Altı tekrarlayan sütun kaldırıldı.
2. `Toplam_Ucret ≈ Aylik_Ucret × Musterilik_Suresi_Ay`. Bu çarpımsal ilişki ikili
   Pearson korelasyonunda (0.83) tam görünmüyordu; katsayının neredeyse +1 çıkması
   ve Cond. No.'nun yüksek kalması ile tespit edildi. `Toplam_Ucret` çıkarıldı.

Kalan multicollinearity katsayıların bireysel yorumunu güvenilmez kılıyor, ancak
tahmin performansını anlamlı şekilde etkilemedi — düzeltme öncesi ve sonrası
confusion matrix neredeyse aynı kaldı.

---

## Bulgular

**Churn'ü artıran faktörler:** fiber optic internet, aylık (month-to-month)
sözleşme, kağıtsız fatura, elektronik çek ile ödeme.

**Koruyucu faktörler:** uzun müşterilik süresi, 1–2 yıllık sözleşme, teknik
destek ve online güvenlik hizmeti.

---

## API Kullanımı

### Örnek 1 — riskli profil

3 aylık müşteri, aylık sözleşme, fiber, elektronik çek, hiç ek hizmet yok:

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5
}
```

```json
{ "ayrilma_olasiligi": 0.9237, "esik": 0.2, "karar": "ayrilir" }
```

### Örnek 2 — sadık profil

60 aylık müşteri, iki yıllık sözleşme, otomatik ödeme, tam hizmet paketi:

```json
{ "ayrilma_olasiligi": 0.0153, "esik": 0.2, "karar": "kalir" }
```

### Python'dan çağırma

```python
import requests

BASE = "https://telcocustomerchurn-87zz.onrender.com"
r = requests.post(f"{BASE}/predict", json=musteri, timeout=90)
print(r.json())
```

Çalışan bir örnek için `src/test_api.py` dosyasına bakın.

### Endpoint'ler

| Yöntem | Yol | Açıklama |
|---|---|---|
| GET | `/health` | Servis durumu, model sürümü, varsayılan eşik |
| POST | `/predict` | Tahmin. `?esik=` ile eşik override edilebilir |
| GET | `/docs` | Swagger arayüzü |

Yanıt ham olasılığı, kullanılan eşiği ve kararı birlikte döndürür — böylece
çağıran taraf kendi eşiğini uygulayabilir.

---

## Lokal Çalıştırma

### Docker ile

```bash
docker build -t telco-api .
docker run -p 7860:7860 telco-api
```

`http://localhost:7860/docs` adresini açın.

### Docker olmadan

```bash
pip install -r requirements.txt
uvicorn src.app:app --reload
```

Modüller göreli import kullandığı için komut proje kökünden çalıştırılmalıdır.

### Modeli yeniden eğitmek

```bash
python -m src.train
```

`pipeline.joblib` ve `meta.json` dosyalarını yeniden üretir.

---

## Üretime Alma Notları

Bu bir portföy servisi. Gerçek bir üretim ortamında ek olarak gerekenler:

- **Kimlik doğrulama** — endpoint şu an herkese açık
- **Rate limiting** — temel bir sınır (`slowapi`, 30 istek/dakika) mevcut, ancak
  dağıtık bir sayaç değil
- **İzleme** — girdi dağılımı kayması ve model performans takibi
- **Model versiyonlama** — artifact ile kod sürümünün birlikte takibi
- **CI/CD** — push öncesi otomatik test

Halihazırda var olan bir koruma: `meta.json` içindeki scikit-learn sürümü ile
çalışma ortamındaki sürüm karşılaştırılıyor, uyuşmazlık varsa servis başlamıyor.
Pickle formatı kütüphane sürümüne duyarlıdır ve uyumsuzluk sessizce yanlış tahmin
üretebilir — bu kontrol onu gürültülü bir hataya çeviriyor.

---

## Teknoloji

Python 3.12 · scikit-learn · imbalanced-learn · pandas · FastAPI · Pydantic ·
Docker · Render

---

## Süreçten Çıkanlar

Modeli notebook'tan çıkarmak, modeli kurmaktan daha öğreticiydi. Notebook'ta
çalışan kod, çalışma dizinine, işletim sistemine ve kurulu kütüphane
sürümlerine dair sessiz varsayımlarla doludur. Konteynerize etmek bu
varsayımların her birini tek tek açığa çıkarıp sabitlemek demek:

- Mutlak import'lar çalışma dizinine bağlıydı; göreli import'a geçildi
- Windows dosya adlarında büyük/küçük harf ayrımı yapmaz, Linux yapar
- Pickle dosyası, üretildiği andaki modül yolunu içeriyordu; paket yapısı
  değişince artifact geçersiz kaldı ve yeniden eğitim gerekti

Bunların hiçbiri model kalitesiyle ilgili değil, ama üçü de modelin başka bir
makinede çalışmasını engelliyordu.
