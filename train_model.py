import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

df = pd.read_csv("data/procurement_data_cleaned.csv")

features = [
    "ordered_qty",
    "delivered_qty",
    "unit_cost",
    "po_value",
    "defect_rate",
    "previous_delay_count",
    "supplier_rating",
    "fill_rate",
    "cost_impact"
]

X = df[features]
y = df["delivery_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model Performance:\n")
print(classification_report(y_test, y_pred))