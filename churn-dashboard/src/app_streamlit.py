
import argparse, joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from preprocess import load_and_clean, split_X_y

st.set_page_config(page_title="Churn Dashboard", layout="wide")

def sidebar_inputs(raw_df):
    st.sidebar.header("Feature Filters")
    selectable = {}
    for col in raw_df.columns:
        if col == 'Churn':
            continue
        if raw_df[col].dtype == 'object' and raw_df[col].nunique() <= 20:
            selectable[col] = st.sidebar.multiselect(col, options=sorted(raw_df[col].dropna().unique()))
        elif pd.api.types.is_numeric_dtype(raw_df[col]):
            min_v, max_v = float(raw_df[col].min()), float(raw_df[col].max())
            selectable[col] = st.sidebar.slider(col, min_v, max_v, (min_v, max_v))
    return selectable

def filter_raw(raw_df, filters):
    df = raw_df.copy()
    for col, val in filters.items():
        if not val:
            continue
        if isinstance(val, list):
            df = df[df[col].isin(val)]
        elif isinstance(val, tuple):
            df = df[(df[col] >= val[0]) & (df[col] <= val[1])]
    return df

def main(args):
    raw = pd.read_csv(args.data)
    df = load_and_clean(args.data)
    X, y = split_X_y(df)

    st.title("Customer Churn Analysis")
    st.caption("Explore patterns and predict churn probability.")

    # Sidebar filters on raw (pre-encoding) to keep UX simple
    filters = sidebar_inputs(raw)
    raw_filtered = filter_raw(raw, filters)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(raw))
    with col2:
        churn_rate = (raw['Churn'].map({'Yes':1,'No':0}).mean())*100
        st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
    with col3:
        st.metric("Filtered Rows", len(raw_filtered))

    # Simple bar chart: Churn by Contract
    if 'Contract' in raw.columns:
        st.subheader("Churn by Contract Type")
        plot_df = raw.groupby(['Contract','Churn']).size().unstack(fill_value=0)
        fig, ax = plt.subplots()
        plot_df.plot(kind='bar', ax=ax)
        ax.set_xlabel("Contract")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Prediction form (requires trained model)
    try:
        clf = joblib.load(args.model)
        st.success("Model loaded. Use prediction with encoded features.")
        st.write("Upload a small CSV (same schema as training data) to predict churn:")
        up = st.file_uploader("CSV file", type=['csv'])
        if up:
            new_raw = pd.read_csv(up)
            # Encode with same pipeline
            new_df = pd.concat([new_raw, pd.Series([0]*len(new_raw), name='Churn')], axis=1)
            enc = load_and_clean  # reuse
            enc_df = enc(new_df.to_csv(index=False))
    except Exception as e:
        st.warning(f"Prediction disabled until model is trained and saved: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=False, default="./models/churn_rf.pkl")
    args = parser.parse_args()
    main(args)
