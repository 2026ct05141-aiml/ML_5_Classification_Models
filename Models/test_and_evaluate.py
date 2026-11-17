import sys
from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing import BoxplotOutlierClipper, DEFAULT_TEST_PATH, RESOURCES_MODULE  # noqa: E402, F401
from Models.classification_models import EXECUTABLE_MODELS_DIR, calculate_metrics  # noqa: E402

EXTERNAL_TEST_PATH = PROJECT_ROOT / "data" / "raintomorrowtest.xlsx"
EXCEL_TABLES_PATH = RESOURCES_MODULE / "evaluation_tables.xlsx"

MODEL_ARTIFACTS = [
    ("Logistic Regression", "logistic_regression.joblib"),
    ("Decision Tree", "decision_tree.joblib"),
    ("kNN", "knn.joblib"),
    ("Naive Bayes", "naive_bayes.joblib"),
    ("Random Forest (Ensemble)", "random_forest.joblib"),
]


def load_saved_bundles() -> dict[str, dict]:
    """Loads all 5 serialized model bundles from Models/executable_models/."""
    bundles = {}
    for model_name, filename in MODEL_ARTIFACTS:
        artifact_path = EXECUTABLE_MODELS_DIR / filename
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Saved model artifact not found at '{artifact_path}'. "
                "Run 'python -m Models.classification_models' first."
            )
        bundles[model_name] = joblib.load(artifact_path)
    print(f"Loaded {len(bundles)} saved model artifacts from '{EXECUTABLE_MODELS_DIR}'.")
    return bundles


def evaluate_on_labeled_test_split(bundles: dict[str, dict], test_csv_path: Path = DEFAULT_TEST_PATH):
    """Evaluates all 5 saved models on the stratified labeled test split (test_data.csv)."""
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Labeled test split not found at '{test_csv_path}'.")

    test_df = pd.read_csv(test_csv_path)
    X_test_raw = test_df.drop(columns=["RainTomorrow"])
    y_test = test_df["RainTomorrow"].astype(int).to_numpy()

    metric_records = []
    predictions = {}

    for model_name, bundle in bundles.items():
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]
        threshold = bundle.get("threshold", 0.50)

        X_test_processed = preprocessor.transform(X_test_raw)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = calculate_metrics(model_name, y_test, y_pred, y_prob)
        metric_records.append(metrics)
        predictions[model_name] = (y_pred, y_prob)

    ordered_cols = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    metrics_df = pd.DataFrame(metric_records)[ordered_cols]

    print("\n" + "=" * 84)
    print("TABLE 1: SAVED MODELS EVALUATION ON LABELED TEST SPLIT (test_data.csv)")
    print("=" * 84)
    print(metrics_df.to_string(index=False))
    print("=" * 84)

    return metrics_df, predictions, y_test


def build_observations_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the structured Model Performance & Observations table."""
    observations = {
        "Logistic Regression": (
            "Strong linear baseline achieving high AUC and balanced MCC/F1 after median imputation, "
            "IQR outlier capping, and standard scaling. Captures additive effects across Humidity3pm, "
            "Sunshine, and barometric pressure drops, though linear boundaries miss non-linear station interactions."
        ),
        "Decision Tree": (
            "Interpretable non-linear baseline with fast inference, but lower AUC and MCC than ensemble "
            "and linear models due to axis-aligned splits across 113 one-hot-encoded features."
        ),
        "kNN": (
            "Distance-weighted neighborhood voting benefits from standard scaling and outlier capping, "
            "delivering high Precision and solid AUC, though 113-dimensional Euclidean distances limit Recall."
        ),
        "Naive Bayes": (
            "Fastest probabilistic model; variance smoothing (1e-1) prevents rare one-hot categories from "
            "collapsing likelihoods, though correlated weather predictors slightly constrain Precision."
        ),
        "Random Forest (Ensemble)": (
            "Best overall model across Accuracy, AUC, Precision, and MCC. Bagging 250 decorrelated trees "
            "captures complex non-linear interactions across stations, humidity, and pressure changes."
        ),
    }

    rows = []
    for _, row in metrics_df.iterrows():
        name = row["ML Model Name"]
        rows.append({
            "ML Model Name": name,
            "Key Metrics": f"Acc={row['Accuracy']:.4f} | AUC={row['AUC']:.4f} | F1={row['F1']:.4f} | MCC={row['MCC']:.4f}",
            "Observation about Model Performance": observations[name],
        })

    obs_df = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print("TABLE 2: MODEL PERFORMANCE OBSERVATIONS")
    print("=" * 84)
    print(obs_df.to_string(index=False))
    print("=" * 84)
    return obs_df


def save_evaluation_visualizations(
    metrics_df: pd.DataFrame,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    y_test: np.ndarray,
):
    """Generates and saves all evaluation charts inside resources/."""
    RESOURCES_MODULE.mkdir(parents=True, exist_ok=True)
    metrics_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3", "#e7298a"]

    # 1. Grouped Bar Chart across all 6 metrics
    x = np.arange(len(metrics_cols))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, (_, row) in enumerate(metrics_df.iterrows()):
        vals = [row[m] for m in metrics_cols]
        offset = (idx - 2) * width
        bars = ax.bar(x + offset, vals, width, label=row["ML Model Name"], color=colors[idx])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Saved Models Comparison Across All 6 Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_cols, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    bar_path = RESOURCES_MODULE / "model_metrics_comparison.png"
    plt.savefig(bar_path, dpi=150)
    plt.close(fig)

    # 2. Combined ROC-AUC Curves
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for idx, (_, row) in enumerate(metrics_df.iterrows()):
        name = row["ML Model Name"]
        _, y_prob = predictions[name]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, color=colors[idx], label=f"{name} (AUC = {row['AUC']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Baseline (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curves Comparison Across All 5 Saved Models", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = RESOURCES_MODULE / "roc_curves_comparison.png"
    plt.savefig(roc_path, dpi=150)
    plt.close(fig)

    # 3. Confusion Matrices Grid
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    for idx, (_, row) in enumerate(metrics_df.iterrows()):
        name = row["ML Model Name"]
        y_pred, _ = predictions[name]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No Rain (0)", "Rain (1)"],
            yticklabels=["No Rain (0)", "Rain (1)"] if idx == 0 else False,
            ax=axes[idx],
        )
        axes[idx].set_title(name, fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        if idx == 0:
            axes[idx].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices Across All 5 Saved Models", fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_path = RESOURCES_MODULE / "confusion_matrices_comparison.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved evaluation charts in '{RESOURCES_MODULE}':")
    print(f"  - {bar_path.name}")
    print(f"  - {roc_path.name}")
    print(f"  - {cm_path.name}")


def predict_on_external_test_set(
    bundles: dict[str, dict],
    external_test_path: Path = EXTERNAL_TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the attached unlabeled external test set (data/raintomorrowtest.xlsx),
    applies the saved training-fitted preprocessor, runs inference across all 5 saved models,
    exports predictions CSV, and saves a prediction distribution chart in resources/.
    """
    if not external_test_path.exists():
        raise FileNotFoundError(
            f"External test file not found at '{external_test_path}'. "
            "Place 'raintomorrowtest.xlsx' inside the 'data/' folder."
        )

    xls = pd.ExcelFile(external_test_path)
    ext_df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    print(f"\nLoaded external test dataset '{external_test_path.name}': {ext_df.shape[0]:,} rows x {ext_df.shape[1]} cols")

    row_ids = ext_df["row ID"] if "row ID" in ext_df.columns else pd.Series(range(len(ext_df)), name="row ID")
    drop_cols = [c for c in ("row ID", "Date", "Temp9am", "Temp3pm", "RainTomorrow") if c in ext_df.columns]
    X_ext_raw = ext_df.drop(columns=drop_cols)

    pred_output = pd.DataFrame({"row ID": row_ids})
    summary_rows = []

    for model_name, bundle in bundles.items():
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]
        threshold = bundle.get("threshold", 0.50)

        X_ext_processed = preprocessor.transform(X_ext_raw)
        y_prob = model.predict_proba(X_ext_processed)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        col_slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        pred_output[f"{col_slug}_pred"] = y_pred
        pred_output[f"{col_slug}_prob"] = np.round(y_prob, 4)

        rain_count = int(y_pred.sum())
        no_rain_count = int(len(y_pred) - rain_count)
        summary_rows.append({
            "ML Model Name": model_name,
            "Predicted No Rain (0)": no_rain_count,
            "Predicted Rain (1)": rain_count,
            "Predicted Rain %": round(rain_count / len(y_pred) * 100, 2),
        })

    # Export full row-level predictions CSV to resources/
    csv_out_path = RESOURCES_MODULE / "raintomorrowtest_predictions.csv"
    pred_output.to_csv(csv_out_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 84)
    print("TABLE 3: EXTERNAL TEST SET (raintomorrowtest.xlsx) PREDICTION SUMMARY")
    print("=" * 84)
    print(summary_df.to_string(index=False))
    print("=" * 84)
    print(f"Saved full external test predictions ({len(pred_output):,} rows) -> '{csv_out_path}'")

    # Visualize predicted class distribution across all 5 models on raintomorrowtest.xlsx
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(summary_df))
    width = 0.35

    bars0 = ax.bar(x - width / 2, summary_df["Predicted No Rain (0)"], width, label="No Rain (0)", color="#3b7ea1")
    bars1 = ax.bar(x + width / 2, summary_df["Predicted Rain (1)"], width, label="Rain (1)", color="#d95f02")

    for bar in list(bars0) + list(bars1):
        h = bar.get_height()
        ax.annotate(
            f"{int(h):,}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Predicted Instance Count", fontsize=11)
    ax.set_title("External Test Set (raintomorrowtest.xlsx - 42,677 Rows) Prediction Distribution", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["ML Model Name"], fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(summary_df["Predicted No Rain (0)"]) * 1.15)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    dist_path = RESOURCES_MODULE / "external_test_predictions_distribution.png"
    plt.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f"Saved external test prediction chart -> '{dist_path}'")

    return pred_output, summary_df


def export_tables_to_excel(
    metrics_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    excel_path: Path = EXCEL_TABLES_PATH,
):
    """Exports all 3 evaluation & observation tables into a formatted multi-sheet Excel workbook."""
    RESOURCES_MODULE.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Evaluation_Metrics", index=False)
        obs_df.to_excel(writer, sheet_name="Model_Observations", index=False)
        summary_df.to_excel(writer, sheet_name="External_Test_Summary", index=False)

        # Auto-fit column widths across all sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for col in worksheet.columns:
                max_len = max((len(str(cell.value)) for cell in col if cell.value is not None), default=10)
                worksheet.column_dimensions[col[0].column_letter].width = min(max_len + 4, 85)

    print(f"\nSaved all evaluation tables to Excel workbook -> '{excel_path}'")


def main():
    bundles = load_saved_bundles()
    metrics_df, predictions, y_test = evaluate_on_labeled_test_split(bundles)
    obs_df = build_observations_table(metrics_df)
    save_evaluation_visualizations(metrics_df, predictions, y_test)
    _, summary_df = predict_on_external_test_set(bundles)
    export_tables_to_excel(metrics_df, obs_df, summary_df)


if __name__ == "__main__":
    main()