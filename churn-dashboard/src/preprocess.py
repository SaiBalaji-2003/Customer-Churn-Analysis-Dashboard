
import pandas as pd
from typing import Tuple

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip spaces in column names
    df.columns = [c.strip() for c in df.columns]
    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Map target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0}).astype('Int64')
    # Drop rows with missing target
    df = df.dropna(subset=['Churn'])
    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    # Fill numerics
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    # One-hot encode categoricals (excluding target)
    cat_cols = [c for c in df.select_dtypes(include='object').columns if c != 'Churn']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df['Churn'].astype(int)
    X = df.drop(columns=['Churn'])
    return X, y
