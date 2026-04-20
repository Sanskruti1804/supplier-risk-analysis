import pandas as pd
import numpy as np
from faker import Faker
from datetime import timedelta

fake = Faker()
np.random.seed(42)

rows = 2000

suppliers = [f"Supplier_{i}" for i in range(1, 51)]
regions = ["Germany", "India", "China", "Poland", "France"]
categories = ["Electronics", "Mechanical", "Raw Material", "Packaging"]

data = []

for i in range(rows):
    supplier = np.random.choice(suppliers)
    region = np.random.choice(regions)
    category = np.random.choice(categories)

    po_date = fake.date_between(start_date='-1y', end_date='today')
    lead_time = np.random.randint(5, 30)
    promised_date = po_date + timedelta(days=lead_time)

    delay = np.random.choice([0, 0, 1, 2, 3, 5, 7])
    actual_date = promised_date + timedelta(days=int(delay))

    qty = np.random.randint(50, 500)
    delivered_qty = qty - np.random.randint(0, 30)
    unit_cost = round(np.random.uniform(10, 200), 2)
    po_value = qty * unit_cost

    defect_rate = round(np.random.uniform(0, 0.1), 3)
    prev_delay = np.random.randint(0, 5)
    rating = round(np.random.uniform(2.5, 5.0), 2)

    data.append([
        f"PO{i}", supplier, region, category,
        po_date, promised_date, actual_date,
        qty, delivered_qty, unit_cost, po_value,
        defect_rate, prev_delay, rating
    ])

columns = [
    "po_id", "supplier_name", "region", "category",
    "po_date", "promised_date", "actual_date",
    "ordered_qty", "delivered_qty", "unit_cost", "po_value",
    "defect_rate", "previous_delay_count", "supplier_rating"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("procurement_data.csv", index=False)

print("Dataset created successfully!")