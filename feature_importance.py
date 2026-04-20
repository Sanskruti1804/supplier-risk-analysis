import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

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

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X, y)

importance = model.feature_importances_

for i, col in enumerate(features):
    print(f"{col}: {importance[i]}")

plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.show()