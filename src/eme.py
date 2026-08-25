# eme.py
from pydantic import ValidationError
from schema import Musteri

ornek = Musteri.model_config["json_schema_extra"]["example"]

senaryolar = [
    ("temiz örnek",        ornek,                                    True),
    ("geçersiz sözleşme",  {**ornek, "Contract": "Three year"},      False),
    ("internet tutarsız",  {**ornek, "InternetService": "No"},       False),
    ("negatif tenure",     {**ornek, "tenure": -5},                  False),
    ("fazla alan",         {**ornek, "Bakiye": 100},                 False),
]

for isim, veri, gecmeli in senaryolar:
    try:
        Musteri(**veri)
        sonuc = True
    except ValidationError:
        sonuc = False
    print(f"{'✅' if sonuc == gecmeli else '❌'} {isim}")