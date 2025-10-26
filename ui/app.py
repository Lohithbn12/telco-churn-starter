import streamlit as st, requests

st.set_page_config(page_title='Telco Churn Demo', page_icon='📉', layout='centered')
st.title('Telco Churn Demo')
st.caption('Enter details and get churn probability.')

API_URL = st.secrets.get('API_URL', 'http://localhost:8000/predict')

defaults = dict(
    gender='Female', SeniorCitizen=0, Partner='Yes', Dependents='No', tenure=12,
    PhoneService='Yes', MultipleLines='No', InternetService='Fiber optic', OnlineSecurity='No',
    OnlineBackup='No', DeviceProtection='No', TechSupport='No', StreamingTV='Yes',
    StreamingMovies='Yes', Contract='Month-to-month', PaperlessBilling='Yes',
    PaymentMethod='Electronic check', MonthlyCharges=70.5, TotalCharges=830.5
)

with st.form('f'):
    cols = list(defaults.keys())
    data = {}
    for k in cols:
        v = defaults[k]
        if isinstance(v, (int, float)):
            data[k] = st.number_input(k, value=float(v))
            if k in ('SeniorCitizen','tenure'):
                data[k] = int(data[k])
        else:
            data[k] = st.text_input(k, value=v)
    submitted = st.form_submit_button('Predict')

if submitted:
    try:
        r = requests.post(API_URL, json=data, timeout=10)
        res = r.json()
        st.success(f"Churn Probability: {res.get('churn_proba', 0):.3f}")
        st.write(res)
    except Exception as e:
        st.error(f'Error contacting API: {e}')
        st.info('Make sure the FastAPI service is running and API_URL is correct.')
