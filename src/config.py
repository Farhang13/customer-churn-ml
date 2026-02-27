import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "telco_churn.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"