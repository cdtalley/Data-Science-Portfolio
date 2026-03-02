"""Pydantic schemas for the churn prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Raw customer features matching the Telecom Churn dataset."""

    gender: str = Field(..., examples=["Male"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[70.35])


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: bool = Field(..., description="Churn prediction at optimal threshold")
    risk_tier: str = Field(..., description="Low / Medium / High risk")
    recommended_action: str
    threshold_used: float
    estimated_cost_if_missed: float = Field(
        ..., description="Estimated cost if this churner is missed ($)"
    )


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchSummary(BaseModel):
    total: int
    predicted_churners: int
    high_risk: int
    medium_risk: int
    low_risk: int
    total_risk_exposure: float = Field(
        ..., description="Sum of estimated costs if all predicted churners are missed"
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: BatchSummary


class ModelInfo(BaseModel):
    model_type: str
    n_features: int
    feature_names: list[str]
    metrics: dict[str, float]
    threshold: float
    cost_assumptions: dict[str, float]
    trained_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str = Field(..., description="'increases_churn' or 'decreases_churn'")


class ExplainResponse(BaseModel):
    churn_probability: float
    top_drivers: list[FeatureContribution]
