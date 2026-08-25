# schema.py
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvetHayir = Literal["Yes", "No"]
InternetliEvetHayir = Literal["Yes", "No", "No internet service"]


class Musteri(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.9,
            }
        },
    )

    # --- demografik ---
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: EvetHayir
    Dependents: EvetHayir

    # --- hesap ---
    tenure: int = Field(ge=0, le=200, description="Müşterilik süresi (ay)")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: EvetHayir
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0, le=1000, description="Aylık ücret")

    # --- hizmetler ---
    PhoneService: EvetHayir
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetliEvetHayir
    OnlineBackup: InternetliEvetHayir
    DeviceProtection: InternetliEvetHayir
    TechSupport: InternetliEvetHayir
    StreamingTV: InternetliEvetHayir
    StreamingMovies: InternetliEvetHayir

    @model_validator(mode="after")
    def hizmet_tutarliligi(self):
        internetli = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        if self.InternetService == "No":
            hatali = [a for a in internetli if getattr(self, a) != "No internet service"]
            if hatali:
                raise ValueError(
                    f"InternetService='No' iken şunlar 'No internet service' olmalı: {hatali}"
                )
        else:
            hatali = [a for a in internetli if getattr(self, a) == "No internet service"]
            if hatali:
                raise ValueError(
                    f"İnternet hizmeti varken şunlar 'No internet service' olamaz: {hatali}"
                )

        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("PhoneService='No' iken MultipleLines='No phone service' olmalı")
        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError("PhoneService='Yes' iken MultipleLines='No phone service' olamaz")

        return self