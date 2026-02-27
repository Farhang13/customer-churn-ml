def create_features(df):
    df = df.copy()

    # Example engineered feature
    df["RevenuePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df