# Customer Churn Prediction System  
### End-to-End Machine Learning Pipeline (Telecom / Fintech Use Case)

---

## Executive Summary

Customer churn is one of the largest revenue leakages in subscription-based businesses such as telecom operators, digital banks, and SaaS platforms.

This project builds a production-style Machine Learning pipeline to:

- Predict churn probability at the customer level
- Identify high-risk customer segments
- Support targeted retention strategies
- Quantify business trade-offs between recall and precision

The final model achieves:

- **ROC-AUC: 0.8472**
- **PR-AUC: 0.6618**
- **Churn detection rate (Recall): 78.34%**

The system is modular, reproducible, and ready for deployment.

---

## 1. Business Context

In recurring-revenue models, losing an existing customer is significantly more expensive than retaining one.

Key business questions addressed:

- Which customers are most likely to churn?
- How many churners can we realistically detect?
- What is the trade-off between retention cost and churn prevention?
- Which customer characteristics drive churn risk?

The objective is not just prediction accuracy, but **actionable churn prevention**.

---

## 2. Dataset Overview

Dataset: Telco Customer Churn  

- ~7,000 customers  
- 20+ behavioral and contractual features  
- Churn rate ≈ 26%

Key variables include:

- Contract type (monthly vs yearly)
- Tenure
- Monthly charges
- Internet service type
- Payment method
- Add-on services

---

## 3. System Architecture

```text
customer-churn-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── churn_model.pkl
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── pipeline.py
│   ├── train.py
│   └── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md
```

The system uses:

- Scikit-Learn Pipelines
- ColumnTransformer (clean separation of numeric & categorical features)
- Class imbalance handling
- Modular training & evaluation workflow
- Persisted model artifact (`churn_model.pkl`)

This structure mirrors real production ML repositories.

---

## 4. Modeling Strategy

### Compared Models

| Model           | ROC-AUC | PR-AUC |
|-----------------|---------|--------|
| Logistic Reg.   | 0.8472  | 0.6618 |
| Random Forest   | 0.8224  | 0.6190 |
| XGBoost         | 0.8350  | 0.6450 |

**Best Model: Logistic Regression**

Despite testing ensemble methods, Logistic Regression achieved the strongest performance, showing that:

- Proper preprocessing is critical
- Class balancing significantly impacts performance
- Simpler models can outperform complex ones
- Interpretability is a competitive advantage

---

## 5. Final Evaluation (Threshold = 0.50)

Confusion Matrix:
[[753 282]
[ 81 293]]


### Performance Metrics

- Accuracy: 74.24%
- Precision: 50.96%
- Recall (Churn Detection Rate): 78.34%
- F1-score: 0.6175
- False Alarm Rate: 27.25%

### Interpretation

- The model detects **78% of churners**
- Roughly **1 in 2 flagged customers is a true churner**
- 27% of retained customers are incorrectly flagged

This configuration prioritizes **high recall**, appropriate when the cost of losing a customer exceeds the cost of contacting them.

---

## 6. Business Implications

### High-Impact Insights

Customers more likely to churn typically include:

- Month-to-month contract holders
- Low-tenure customers
- High monthly charge customers
- Specific service-type segments

### Operational Strategy

1. Rank customers by predicted churn probability
2. Target the top-risk segment (e.g., top 10–20%)
3. Offer contract incentives or tailored retention offers
4. Improve onboarding for new customers

---

## 7. Business Impact Simulation

Assume:

- Average monthly revenue per customer: €40
- Retention campaign cost: €10 per contacted customer
- Model recall: 78%

If the company retains just 20% of predicted churners:

The net financial gain could substantially exceed retention campaign costs.

This demonstrates that predictive modeling can directly support revenue optimization.

---

## 8. Why This Approach Is Industry-Ready

- Modular, reproducible codebase
- No data leakage (pipeline-based preprocessing)
- Class imbalance handled explicitly
- Clear evaluation framework
- Business-aligned metrics
- Interpretable model (important for GDPR compliance in Europe)

This makes the system suitable for banking, fintech, telecom, and subscription-based businesses.

---

