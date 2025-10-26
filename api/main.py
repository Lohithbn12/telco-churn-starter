from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib, os

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model.joblib')
pipe = joblib.load(MODEL_PATH)

app = FastAPI(title='Telco Churn API', version='1.0')

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(payload: Customer):
    x = [payload.model_dump()]
    proba = pipe.predict_proba(x)[:,1][0]
    pred = int(proba >= 0.5)
    return {'churn_proba': float(proba), 'churn_pred': pred}
