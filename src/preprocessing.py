import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop ID column (not predictive, can cause leakage-ish behavior)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges: convert blanks to NaN then numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Strip whitespace from all string columns (common hygiene)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Map target to 0/1
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    return df