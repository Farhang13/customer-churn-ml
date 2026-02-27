import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from .config import MODEL_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from .pipeline import build_preprocessor


def train_and_compare(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(df, TARGET_COLUMN)

    models = {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )
    }

    results = []
    fitted = {}

    for name, model in models.items():
        clf = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)

        results.append((name, roc, pr_auc))
        fitted[name] = clf

        print(f"{name:12s} | ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")

    # pick best by ROC-AUC (you can also pick by PR-AUC)
    best_name, best_roc, best_pr = sorted(results, key=lambda x: x[1], reverse=True)[0]
    best_model = fitted[best_name]

    print("\nBest model:", best_name, "| ROC-AUC:", round(best_roc, 4), "| PR-AUC:", round(best_pr, 4))

    joblib.dump(best_model, MODEL_PATH)
    print("Saved model to:", MODEL_PATH)

    return best_model, (X_test, y_test), results