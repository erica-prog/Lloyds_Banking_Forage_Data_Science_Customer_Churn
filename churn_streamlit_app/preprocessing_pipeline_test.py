"""
preprocessing_pipeline.py
---------------------------------
End‑to‑end preprocessing pipeline for the Lloyds churn project.

What it does
-------------
1. Load raw Excel sheets (Demographics, Transactions, Service, Online Activity, Churn).
2. Aggregate one‑to‑many tables to customer level (sum/mean/count etc.).
3. Engineer features (recency, resolution_rate, etc.).
4. Merge everything into a single modelling table.
5. Split into train/val/test (stratified).
6. Build a scikit‑learn/imbalanced‑learn Pipeline that:
    - imputes missing values,
    - encodes categoricals (OneHot),
    - scales numerics,
    - optionally applies SMOTE only to the training fold.

Usage
-----
python preprocessing_pipeline.py --excel /path/Customer_Churn_Data_Large.xlsx \\
                                 --target ChurnStatus \\
                                 --outdir ./data_processed

The script will save:
    - train_X.csv, train_y.csv
    - val_X.csv,   val_y.csv
    - test_X.csv,  test_y.csv
    - preprocessor.joblib  (fitted ColumnTransformer)
    - feature_names.json   (ordered feature names after OHE)
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


# -------------------------------
# 1. Loading helpers
# -------------------------------
SHEET_MAP = {
    "demographics": "Customer_Demographics",
    "transactions": "Transaction_History",
    "service": "Customer_Service",
    "online": "Online_Activity",
    "churn": "Churn_Status"
}


def load_sheets(excel_path: str) -> dict:
    xl = pd.ExcelFile(excel_path)
    dfs = {k: xl.parse(v) for k, v in SHEET_MAP.items()}
    return dfs


# -------------------------------
# 2. Feature engineering helpers
# -------------------------------
def agg_transactions(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby('CustomerID')['AmountSpent']
    out = pd.DataFrame({
        'CustomerID': g.sum().index,
        'TotalAmountSpent': g.sum().values,
        'AverageAmountSpent': g.mean().values,
        'NumTransactions': g.count().values
    })
    return out


def agg_service(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['InteractionDate'] = pd.to_datetime(df['InteractionDate'])
    grp = df.groupby('CustomerID')
    out = grp.agg(
        number_of_interactions=('InteractionType', 'count'),
        last_interaction=('InteractionDate', 'max')
    ).reset_index()

    resolved = df[df['ResolutionStatus'] == 'Resolved'].groupby('CustomerID').size()
    out['resolved_interactions'] = out['CustomerID'].map(resolved).fillna(0).astype(int)
    out['resolution_rate'] = out['resolved_interactions'] / out['number_of_interactions']
    return out


def engineer_recency(df_online: pd.DataFrame) -> pd.DataFrame:
    df = df_online.copy()
    df['LastLoginDate'] = pd.to_datetime(df['LastLoginDate'])
    max_date = df['LastLoginDate'].max()
    df['Recency'] = (max_date - df['LastLoginDate']).dt.days
    df = df.sort_values('LastLoginDate').groupby('CustomerID').tail(1)
    return df[['CustomerID', 'ServiceUsage', 'LoginFrequency', 'Recency', 'LastLoginDate']]


def merge_all(dfs: dict) -> pd.DataFrame:
    demo = dfs['demographics']
    churn = dfs['churn']

    tx = agg_transactions(dfs['transactions'])
    svc = agg_service(dfs['service'])
    online = engineer_recency(dfs['online'])

    merged = (demo.merge(tx, on='CustomerID', how='left')
                   .merge(svc, on='CustomerID', how='left')
                   .merge(online, on='CustomerID', how='left')
                   .merge(churn, on='CustomerID', how='left'))

    return merged


# -------------------------------
# 3. Preprocessing pipeline
# -------------------------------
def build_preprocessor(df: pd.DataFrame, target: str) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=['number', 'datetime64[ns]']).columns.tolist()
    for col in ['CustomerID', 'LastLoginDate', 'last_interaction']:
        if col in num_cols:
            num_cols.remove(col)

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, num_cols),
            ('cat', categorical_pipe, cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor, num_cols, cat_cols


# -------------------------------
# 4. Train/Val/Test split
# -------------------------------
def stratified_splits(df: pd.DataFrame, target: str,
                      test_size: float = 0.2,
                      val_size: float = 0.1,
                      random_state: int = 42):
    y = df[target]
    X = df.drop(columns=[target])
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_ratio, random_state=random_state, stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# -------------------------------
# 5. Build imblearn pipeline (SMOTE optional)
# -------------------------------
def build_training_pipeline(preprocessor: ColumnTransformer,
                            apply_smote: bool = True) -> ImbPipeline:
    steps = [('preprocess', preprocessor)]
    if apply_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    pipe = ImbPipeline(steps=steps)
    return pipe


# -------------------------------
# 6. Main CLI
# -------------------------------
def main(args):
    dfs = load_sheets(args.excel)
    df = merge_all(dfs)

    df = df.dropna(subset=[args.target]).reset_index(drop=True)

    preprocessor, num_cols, cat_cols = build_preprocessor(df, args.target)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(df, args.target,
                                                                       test_size=args.test_size,
                                                                       val_size=args.val_size,
                                                                       random_state=args.seed)

    preprocessor.fit(X_train)

    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_feature_names
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def transform_and_save(X, y, prefix):
        X_t = preprocessor.transform(X)
        X_df = pd.DataFrame(X_t, columns=feature_names, index=X.index)
        X_df.to_csv(outdir / f"{prefix}_X.csv", index=False)
        y.to_csv(outdir / f"{prefix}_y.csv", index=False)

    transform_and_save(X_train, y_train, 'train')
    transform_and_save(X_val, y_val, 'val')
    transform_and_save(X_test, y_test, 'test')

    joblib.dump(preprocessor, outdir / 'preprocessor.joblib')
    with open(outdir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    print(f"Saved processed splits and artifacts to: {outdir.resolve()}\n")
    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)
    print("Total transformed features:", len(feature_names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Lloyds churn data.")
    parser.add_argument(
    '--excel',
    type=str,
    default=r"C:\Users\erika\OneDrive\Desktop\churn_streamlit_app\Customer_Churn_Data_Large.xlsx",
    help='Path to Customer_Churn_Data_Large.xlsx')
    parser.add_argument('--target', type=str, default='ChurnStatus')
    parser.add_argument('--outdir', type=str, default='./data_processed')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
