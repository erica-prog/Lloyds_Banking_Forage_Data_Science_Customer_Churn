
# =============================================================================
# train_selected_models.py
# ------------------------
# Train Logistic Regression (lr), Random Forest (rf), Decision Tree (dt),
# and a simple Neural Network (nn) on the **preprocessed** churn data.
# 
# Input  (from preprocessing_pipeline.py):
#     data_dir/
#         train_X.csv  train_y.csv
#         val_X.csv    val_y.csv
#         test_X.csv   test_y.csv
# 
# Output (per model, in out_dir/):
#     <name>_pipeline.joblib   # sklearn/imb pipelines
#     <name>_pipeline.pkl
#     <name>_report_val.txt
#     <name>_report_test.txt
#     <name>_roc.png           # individual ROC curve
# 
# For nn:
#     nn_model.h5              # Keras saved model
#     nn_scaler.joblib         # StandardScaler used (if any)
#     nn_report_*.txt
#     nn_roc.png
# 
# Also:
#     metrics_summary.csv      # one-line metrics for all models
#     roc_curves_all.png       # combined ROC curves (test set)
# 
# Usage
# -----
# python train_selected_models.py --data_dir "C:\Users\erika\OneDrive\Desktop\churn_streamlit_ap\data_processed" --out_dir models_lr_rf_dt_nn
# =============================================================================


import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             average_precision_score, confusion_matrix)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


def load_splits(data_dir: str):
    data_dir = Path(data_dir)
    X_train = pd.read_csv(os.path.join(data_dir, "train_X.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "train_y.csv")).iloc[:, 0]
    X_val   = pd.read_csv(os.path.join(data_dir, "val_X.csv"))
    y_val   = pd.read_csv(os.path.join(data_dir, "val_y.csv")).iloc[:, 0]
    X_test  = pd.read_csv(os.path.join(data_dir, "test_X.csv"))
    y_test  = pd.read_csv(os.path.join(data_dir, "test_y.csv")).iloc[:, 0]
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(pipe, X, y) -> Dict[str, object]:
    proba = pipe.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    report = classification_report(y, pred, digits=4)
    roc_auc = roc_auc_score(y, proba)
    pr_auc  = average_precision_score(y, proba)
    cm = confusion_matrix(y, pred)
    fpr, tpr, _ = roc_curve(y, proba)
    return {"report": report, "roc_auc": roc_auc, "pr_auc": pr_auc,
            "cm": cm, "fpr": fpr, "tpr": tpr, "proba": proba}


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def plot_single_roc(fpr, tpr, auc, out_path: Path, label: str):
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.2f})")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(args.data_dir)

    # ------------- SKLEARN MODELS -------------
    models = {
        "lr": LogisticRegression(max_iter=1000, random_state=args.seed),
        "rf": RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1),
        "dt": DecisionTreeClassifier(random_state=args.seed),
        
    }
    if XGBClassifier is not None:
        models["xgb"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1
        )

    metrics_rows = []
    roc_all = {}

    for name, estimator in models.items():
        print(f"\n=== Training {name} ===")
        pipe = Pipeline([("smote", SMOTE(random_state=args.seed)),
                         ("clf", estimator)])
        pipe.fit(X_train, y_train)

        # eval
        val_res  = evaluate(pipe, X_val,  y_val)
        test_res = evaluate(pipe, X_test, y_test)

        # save reports
        (out_dir / f"{name}_report_val.txt").write_text(val_res["report"])
        (out_dir / f"{name}_report_test.txt").write_text(test_res["report"])

        # save model
        import joblib
        joblib.dump(pipe, out_dir / f"{name}_pipeline.joblib")
        save_pickle(pipe, out_dir / f"{name}_pipeline.pkl")
        print(f"Saved {name}_pipeline.joblib & .pkl")

        # ROC plots
        plot_single_roc(test_res["fpr"], test_res["tpr"], test_res["roc_auc"],
                        out_dir / f"{name}_roc.png", name.upper())

        metrics_rows.append({
            "model": name,
            "roc_auc_val":  val_res["roc_auc"],
            "pr_auc_val":   val_res["pr_auc"],
            "roc_auc_test": test_res["roc_auc"],
            "pr_auc_test":  test_res["pr_auc"],
            "accuracy_test": (test_res["cm"].trace() / test_res["cm"].sum())
        })
        roc_all[name] = (test_res["fpr"], test_res["tpr"], test_res["roc_auc"])

    # ------------- NEURAL NETWORK -------------
    if args.nn:
        try:
            import tensorflow as tf
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            print(f"TensorFlow version: {tf.__version__}")
            
            # Configure TensorFlow (especially for GPU memory)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Found {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
            
            print("\n=== Training Neural Network (Keras) ===")
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s   = scaler.transform(X_val)
            X_test_s  = scaler.transform(X_test)
    
            # SMOTE after scaling
            sm = SMOTE(random_state=args.seed)
            X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
            
            print(f"Training data shape after SMOTE: {X_train_bal.shape}")
            print(f"Class distribution after SMOTE: {np.bincount(y_train_bal)}")
    
            model = Sequential([
                Dense(64, activation="relu", input_shape=(X_train_bal.shape[1],)),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid")
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss="binary_crossentropy",
                          metrics=["accuracy"])
            
            print("Model compiled successfully")
            print(f"Model summary:")
            model.summary()
            
            # Add early stopping to prevent overfitting
            from tensorflow.keras.callbacks import EarlyStopping
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            print("Starting training...")
            history = model.fit(X_train_bal, y_train_bal,
                                epochs=args.epochs, 
                                batch_size=32,
                                validation_data=(X_val_s, y_val),
                                callbacks=[early_stop],
                                verbose=1)  # Changed to 1 to see progress
    
            # evaluate
            print("Evaluating model...")
            y_prob = model.predict(X_test_s, verbose=0).ravel()
            y_pred = (y_prob >= 0.5).astype(int)
            report = classification_report(y_test, y_pred, digits=4)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc  = average_precision_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
    
            # save stuff
            model.save(out_dir / "nn_model.h5")
            import joblib
            joblib.dump(scaler, out_dir / "nn_scaler.joblib")
            (out_dir / "nn_report_test.txt").write_text(report)
            plot_single_roc(fpr, tpr, roc_auc, out_dir / "nn_roc.png", "NN")
    
            metrics_rows.append({
                "model": "nn",
                "roc_auc_val":  np.nan,
                "pr_auc_val":   np.nan,
                "roc_auc_test": roc_auc,
                "pr_auc_test":  pr_auc,
                "accuracy_test": (cm.trace() / cm.sum())
            })
            roc_all["nn"] = (fpr, tpr, roc_auc)
            
            print("Neural network training completed successfully!")
            
        except ImportError as e:
            print(f"TensorFlow import error: {e}")
            print("Please install TensorFlow: pip install tensorflow")
        except Exception as e:
            print(f"Neural network training failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

    # ------------- Summary -------------
    df_metrics = pd.DataFrame(metrics_rows).sort_values("roc_auc_test", ascending=False)
    df_metrics.to_csv(out_dir / "metrics_summary.csv", index=False)

    # combined ROC
    plt.figure(figsize=(7,6))
    for name, (fpr, tpr, auc_) in roc_all.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name.upper()} (AUC={auc_:.2f})")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves_all.png", dpi=200)
    plt.close()

    print("\nDone. See metrics_summary.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\Users\erika\OneDrive\Desktop\churn_streamlit_app\data_processed",
        help="Folder with train_X.csv etc."
    )
    p.add_argument("--out_dir",  type=str, default="models_lr_rf_dt_nn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nn", action="store_true", help="Train the Keras neural network too")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()
    main(args)
