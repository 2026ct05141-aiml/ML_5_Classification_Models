import sys
from pathlib import Path

import joblib

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
) 

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing import split_train_test_data  

EXECUTABLE_MODELS_DIR = Path(__file__).resolve().parent / "executable_models"
EXECUTABLE_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _find_optimal_mcc_threshold(clf, X_train: np.ndarray, y_train: np.ndarray) -> float:

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    clf.fit(X_tr, y_tr)
    val_probs = clf.predict_proba(X_val)[:, 1]

    best_thresh = 0.50
    best_mcc = -1.0
    for thresh in np.arange(0.38, 0.56, 0.02):
        preds = (val_probs >= thresh).astype(int)
        score = matthews_corrcoef(y_val, preds)
        if score > best_mcc:
            best_mcc = score
            best_thresh = float(round(thresh, 2))
    return best_thresh

def calculate_metrics(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    
    return {
        "ML Model Name": model_name,
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "Precision": round(float(precision_score(y_true, y_pred)), 4),
        "Recall": round(float(recall_score(y_true, y_pred)), 4),
        "F1": round(float(f1_score(y_true, y_pred)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def run_logistic_regression(X_train, y_train, X_test, y_test, preprocessor) -> dict:

    clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    best_thresh = _find_optimal_mcc_threshold(clf, X_train, y_train)

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = calculate_metrics("Logistic Regression", y_test, y_pred, y_prob)
    save_path = EXECUTABLE_MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(
        {"model": clf, "preprocessor": preprocessor, "threshold": best_thresh, "metrics": metrics},
        save_path,
    )
    print(f"[Logistic Regression] Saved -> '{save_path}' (threshold={best_thresh:.2f})")
    return metrics

def run_decision_tree(X_train, y_train, X_test, y_test, preprocessor) -> dict:

    clf = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=40,
        min_samples_leaf=25,
        random_state=42,
    )
    best_thresh = _find_optimal_mcc_threshold(clf, X_train, y_train)

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = calculate_metrics("Decision Tree", y_test, y_pred, y_prob)
    save_path = EXECUTABLE_MODELS_DIR / "decision_tree.joblib"
    joblib.dump(
        {"model": clf, "preprocessor": preprocessor, "threshold": best_thresh, "metrics": metrics},
        save_path,
    )
    print(f"[Decision Tree]       Saved -> '{save_path}' (threshold={best_thresh:.2f})")
    return metrics


def run_knn(X_train, y_train, X_test, y_test, preprocessor) -> dict:

    clf = KNeighborsClassifier(
        n_neighbors=25,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )
    best_thresh = _find_optimal_mcc_threshold(clf, X_train, y_train)

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = calculate_metrics("kNN", y_test, y_pred, y_prob)
    save_path = EXECUTABLE_MODELS_DIR / "knn.joblib"
    joblib.dump(
        {"model": clf, "preprocessor": preprocessor, "threshold": best_thresh, "metrics": metrics},
        save_path,
    )
    print(f"[kNN]                 Saved -> '{save_path}' (threshold={best_thresh:.2f})")
    return metrics


def run_naive_bayes(X_train, y_train, X_test, y_test, preprocessor) -> dict:

    clf = GaussianNB(var_smoothing=1e-1)
    best_thresh = _find_optimal_mcc_threshold(clf, X_train, y_train)

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = calculate_metrics("Naive Bayes", y_test, y_pred, y_prob)
    save_path = EXECUTABLE_MODELS_DIR / "naive_bayes.joblib"
    joblib.dump(
        {"model": clf, "preprocessor": preprocessor, "threshold": best_thresh, "metrics": metrics},
        save_path,
    )
    print(f"[Naive Bayes]         Saved -> '{save_path}' (threshold={best_thresh:.2f})")
    return metrics


def run_random_forest(X_train, y_train, X_test, y_test, preprocessor) -> dict:

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    best_thresh = _find_optimal_mcc_threshold(clf, X_train, y_train)

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = calculate_metrics("Random Forest (Ensemble)", y_test, y_pred, y_prob)
    save_path = EXECUTABLE_MODELS_DIR / "random_forest.joblib"
    joblib.dump(
        {"model": clf, "preprocessor": preprocessor, "threshold": best_thresh, "metrics": metrics},
        save_path,
    )
    print(f"[Random Forest]       Saved -> '{save_path}' (threshold={best_thresh:.2f})")
    return metrics


def train_classification_models() -> pd.DataFrame:
    
    X_train, X_test, y_train, y_test, preprocessor, _ = split_train_test_data()

    results = [
        run_logistic_regression(X_train, y_train, X_test, y_test, preprocessor),
        run_decision_tree(X_train, y_train, X_test, y_test, preprocessor),
        run_knn(X_train, y_train, X_test, y_test, preprocessor),
        run_naive_bayes(X_train, y_train, X_test, y_test, preprocessor),
        run_random_forest(X_train, y_train, X_test, y_test, preprocessor),
    ]

    ordered_cols = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    metrics_data = pd.DataFrame(results)[ordered_cols]

    print("\n" + "=" * 82)
    print("TRAINING & HOLDOUT SPLIT EVALUATION METRICS (ALL 5 MODELS)")
    print("=" * 82)
    print(metrics_data.to_string(index=False))
    print("=" * 82)

    best_row = metrics_data.loc[metrics_data["MCC"].idxmax()]
    print(
        f"\nBest Overall Model (by MCC & AUC): {best_row['ML Model Name']} "
        f"(MCC={best_row['MCC']:.4f}, AUC={best_row['AUC']:.4f}, Accuracy={best_row['Accuracy']:.4f})"
    )
    return metrics_data


if __name__ == "__main__":
    train_classification_models()