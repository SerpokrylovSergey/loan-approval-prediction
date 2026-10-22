from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Сервис кредитного скоринга на базе CatBoost",
    version="1.0.0",
)

model = joblib.load("models/catboost_loan_model.joblib")


class LoanApplication(BaseModel):
  no_of_dependents: int = 2
  education: str = "Graduate"
  self_employed: str = "No"
  income_annum: int = 9600000
  loan_amount: int = 29900000
  loan_term: int = 12
  cibil_score: int = 778
  residential_assets_value: int = 2400000
  commercial_assets_value: int = 17600000
  luxury_assets_value: int = 22700000
  bank_asset_value: int = 8000000


@app.post("/predict")
def predict_loan_status(application: LoanApplication):
  input_df = pd.DataFrame([application.model_dump()])

  prediction = int(model.predict(input_df)[0])
  probability = float(model.predict_proba(input_df)[0][1])

  return {
      "loan_approved": bool(prediction == 1),
      "status": "Approved" if prediction == 1 else "Rejected",
      "approval_probability": round(probability, 4),
  }


@app.get("/health")
def health_check():
  return {"status": "ok"}