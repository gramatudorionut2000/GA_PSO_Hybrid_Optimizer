import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .api import quick_optimize
import logging
import sys
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

df = pd.read_csv("/home/superuser/Desktop/customer_churn_dataset/archive/customer_churn_dataset.csv")

y = df["churn"]
X_df = df.drop(columns=["churn"])

id_cols = [c for c in X_df.columns if 'id' in c.lower()]
X_df = X_df.drop(columns=id_cols, errors='ignore')

for col in X_df.select_dtypes(include=['object', 'category']).columns:
    X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y.astype(str))

X = X_df.values.astype(np.float64)

result = quick_optimize(
    X, y,  # pyright: ignore[reportArgumentType]
    model="gradient_boosting",
    preset="fast",
    task="classification",
    scoring="roc_auc",
    verbose=2
)

print(result.summary())