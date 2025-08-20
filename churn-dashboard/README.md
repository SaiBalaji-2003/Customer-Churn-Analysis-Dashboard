
# Customer Churn Analysis Dashboard

**Goal:** Build a churn prediction pipeline and a lightweight dashboard to explore churn drivers.

## Dataset
Use the Telco Customer Churn dataset (Kaggle): https://www.kaggle.com/blastchar/telco-customer-churn  
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `./data/`

## Run
```bash
# Train model
python src/train_model.py --data ./data/WA_Fn-UseC_-Telco-Customer-Churn.csv --model ./models/churn_rf.pkl

# Generate evaluation report
python src/train_model.py --data ./data/WA_Fn-UseC_-Telco-Customer-Churn.csv --model ./models/churn_rf.pkl --report ./reports/eval.txt

# Launch Streamlit dashboard
streamlit run src/app_streamlit.py -- --data ./data/WA_Fn-UseC_-Telco-Customer-Churn.csv --model ./models/churn_rf.pkl
```

## Files
- `src/preprocess.py` — clean & encode data
- `src/train_model.py` — train RandomForest and save model
- `src/app_streamlit.py` — quick dashboard to inspect churn patterns and make predictions
