# Telco Churn — End-to-End ML (Train → API → UI → Docker → Deploy)

## 1) Quickstart

```bash
# Create venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Train (reads the CSV in data/ and writes model to models/)
python src/train.py

# Run API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Try a request (new terminal)
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
 "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,
 "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No",
 "OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
 "StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
 "MonthlyCharges":70.5,"TotalCharges":830.5
}'
```

## 2) Streamlit UI

```bash
# With API running (above)
streamlit run ui/app.py
```

You can change the API URL by adding to `.streamlit/secrets.toml`:
```
API_URL = "https://your-cloud-run-or-render-url/predict"
```

## 3) Docker

```bash
docker build -t telco-churn:latest .
docker run -p 8000:8000 telco-churn:latest
```

## 4) Deploy (Cloud Run)

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/telco-churn
gcloud run deploy telco-churn --image gcr.io/PROJECT_ID/telco-churn --platform managed --region asia-south1 --allow-unauthenticated --port 8000
```

## Notes
- Training code saves metrics to `models/metrics.json` and the pipeline+model to `models/model.joblib`.
- The API loads the pipeline directly; the preprocessor is embedded.
- For batch predictions: `python src/predict_batch.py --in your_features.csv --model models/model.joblib --out predictions.csv`.
