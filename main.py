from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.train import train_and_compare
from src.evaluate import evaluate_at_threshold

def main():
    df = load_data()
    df = clean_data(df)
    df = create_features(df)

    best_model, (X_test, y_test), results = train_and_compare(df)

    # Predict probabilities for churn (class 1)
    probs = best_model.predict_proba(X_test)[:, 1]

    # Evaluate at default threshold 0.50
    evaluate_at_threshold(y_test, probs, threshold=0.50)

if __name__ == "__main__":
    main()