# Telco Customer Churn Prediction

Bir telekom şirketinin müşteri verilerini kullanarak, hangi müşterilerin şirketten
ayrılma (churn) riski taşıdığını tahmin eden bir sınıflandırma modeli.
model hem klasik yöntem hemde pipeline methodu ile çalışıyor bunun sebebi ikisinide
karşılaştırmak projeyi başlattıkdan sonra 1 derseniz pipeline methoduyla 
2 derseniz eğer klasik yöntem ile çalışacaktır, pipeline methodunda çalışması için
**Scikit-Learn** kütüphanesine uygun transformer sınıfı yazıldı prepare.py, unit test eklendi TestPrepareClass.py.

## Problem Tanımı

**Hedef:** Müşteri özelliklerine (sözleşme tipi, hizmetler, ödeme yöntemi, müşterilik
süresi vb.) bakarak, o müşterinin gelecekte şirketten ayrılıp ayrılmayacağını tahmin
etmek.

**İş değeri:** Yeni müşteri kazanmak, mevcut müşteriyi elde tutmaktan çok daha
pahalıdır. Ayrılma riski taşıyan müşterileri önceden tespit edebilen bir şirket,
onlara özel kampanya/indirim sunarak kaybı önleyebilir.

**Veri seti:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7043 müşteri, 21 özellik.

## Veri Temizliği

- Sütun isimleri anlamlı Türkçe isimlere çevrildi.
- `Toplam_Ucret` (TotalCharges) sütunu görünüşte sayısal olmasına rağmen `str` tipinde
  geliyordu. İncelemede, henüz bir ay bile doldurmamış (`tenure=0`) 11 müşterinin bu
  alanının boşluk karakteriyle dolu olduğu tespit edildi. Bu satırlar `0` ile
  dolduruldu — mantık: yeni müşterinin henüz toplam ödemesi oluşmamıştır.

## Encoding

- İkili (Yes/No, Female/Male) sütunlar `map()` ile 0/1'e çevrildi.
- Çok kategorili sütunlar (Sözleşme Tipi, Ödeme Yöntemi, İnternet Servis Tipi vb.)
  `pd.get_dummies(..., drop_first=True)` ile one-hot encode edildi.
  `drop_first=True` kullanılmasının sebebi **dummy variable trap**'i önlemek: bir
  kategorik grubun tüm seviyeleri tutulursa, seviyelerin toplamı her satırda sabit
  (1) olduğundan, sabit terimle birlikte matematiksel bir bağımlılık oluşur.

## Multicollinearity Tespiti ve Düzeltmesi

Model kurulduktan sonra `statsmodels` ile OLS regresyon tablosu incelendi ve
`Cond. No.` değerinin aşırı yüksek (8.37e+17) çıktığı, en küçük özdeğerin sıfıra
çok yakın olduğu görüldü — bu, tasarım matrisinde güçlü bir doğrusal bağımlılık
olduğuna işaret ediyordu.

**Tespit edilen sorunlar:**
1. `OnlineSecurity`, `OnlineBackup`, `TechSupport` gibi 7 farklı sütunun
   `"No internet service"` kategorisi, internet hizmeti olmayan müşterilerde
   **birebir aynı bilgiyi** taşıyordu (hepsi aynı katsayıyı alıyordu). 6 tekrarlayan
   sütun kaldırıldı, tek temsilci (`Internet_Servis_Tipi_No`) bırakıldı.
2. `Toplam_Ucret`, `Aylik_Ucret` ve `Musterilik_Suresi_Ay`'ın **çarpımına** yakın bir
   ilişki taşıyordu (`Toplam_Ucret ≈ Aylik_Ucret × Musterilik_Suresi_Ay`). Bu
   çarpımsal ilişki, ikili Pearson korelasyonunda (0.83) tam yansımadığı için
   ilk bakışta gözden kaçtı; katsayının neredeyse tam +1 çıkması ve Cond. No.'nun
   yüksek kalmaya devam etmesiyle doğrulandı. `Toplam_Ucret` modelden çıkarıldı.

**Not:** Kalan multicollinearity'nin büyük kısmı katsayıların *bireysel yorumunu*
güvenilmez kılıyor olsa da, modelin *tahmin performansını* anlamlı şekilde
etkilemedi (Confusion Matrix sonuçları düzeltme öncesi/sonrası neredeyse aynı
kaldı). Feature importance yorumları bu sınırlama göz önünde tutularak okunmalı.

## Dengesiz Veri (Imbalanced Data) Problemi

Hedef sütun dengesiz: müşterilerin ~%73'ü kalıyor, ~%27'si ayrılıyor. Bu durumda
"herkese kalacak" diyen aptal bir model bile ~%73 accuracy alabilir — bu yüzden
accuracy tek başına yanıltıcı, **Recall** ve **Precision**'a (özellikle churn=1
sınıfı için) odaklanıldı.

Üç yaklaşım karşılaştırıldı:

| Model | FN (kaçırılan) | FP (yanlış alarm) | Recall (1) | Precision (1) | Accuracy |
|---|---|---|---|---|---|
| Varsayılan | 150 | 103 | 0.60 | 0.68 | 0.82 |
| `class_weight='balanced'` | 66 | 290 | 0.82 | 0.51 | 0.75 |
| SMOTE (train'e uygulanmış) | 62 | 284 | 0.83 | 0.52 | 0.75 |

**Karar:** Projenin amacı ayrılacak müşterileri mümkün olduğunca kaçırmamak
olduğundan (kaçırılan müşteri, yanlış alarma göre çok daha maliyetli — geri
kazanılamıyor), Recall'ü önceliklendiren `class_weight='balanced'` / SMOTE
modelleri tercih edildi. SMOTE marjinal olarak daha iyi sonuç verse de fark
küçük; büyük veri setlerinde bu farkın daha belirgin olması beklenir.

SMOTE'un yalnızca **train** setine uygulandığına, test setinin gerçek (dengesiz)
dağılımı koruduğuna dikkat edildi — aksi halde test seti gerçek dünyayı temsil
etmeyen, yanıltıcı bir sınav olurdu.

## Sonuçlar

- **En etkili churn sebepleri (pozitif katsayılar):** Fiber optic internet hizmeti,
  aylık (month-to-month) sözleşme, kağıtsız fatura, elektronik çek ile ödeme.
- **En güçlü koruyucu faktörler (negatif katsayılar):** Uzun müşterilik süresi,
  1-2 yıllık sözleşme, teknik destek/online güvenlik hizmeti alıyor olmak.
## Kullanılan Araçlar

Python, pandas, NumPy, scikit-learn (LogisticRegression, StandardScaler,
train_test_split), statsmodels (OLS tanı analizi), imbalanced-learn (SMOTE),
matplotlib.

## Nasıl Çalıştırılır

```bash
python main.py
```

`WA_Fn-UseC_-Telco-Customer-Churn.csv` dosyasının proje klasöründe olması gerekir.

## Öğrenilenler

Bu proje, sadece bir model kurup accuracy raporlamanın ötesinde, **modelin neden
o sonuçları verdiğini sorgulamanın** önemini gösterdi: yüksek görünen bir accuracy
dengesiz veri yüzünden yanıltıcı olabiliyor; katsayı yorumları multicollinearity
yüzünden güvenilmez hale gelebiliyor. Bu proje boyunca "kod çalıştı, sonuç geldi"
ile yetinmek yerine her sonucu sorgulamak, projenin en değerli kısmı oldu.
