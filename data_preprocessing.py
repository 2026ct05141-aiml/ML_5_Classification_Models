from pathlib import Path

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plot
from matplotlib.cbook import boxplot_stats
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "australiaraintomorrowtrain.xlsx"
DEFAULT_TEST_PATH = PROJECT_ROOT / "test_data.csv"
RESOURCES_MODULE = PROJECT_ROOT / "resources"


class BoxplotOutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, whisker: float = 1.5):
        self.whisker = whisker

    def fit(self, X, y=None):
        X_real = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X_real, 25, axis=0)
        q3 = np.nanpercentile(X_real, 75, axis=0)
        iqr = q3 - q1
        self.lower_limit = q1 - self.whisker * iqr
        self.upper_limit = q3 + self.whisker * iqr
        return self

    def transform(self, X):
        X_real = np.asarray(X, dtype=float).copy()
        return np.clip(X_real, self.lower_limit, self.upper_limit)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _ensure_resources_module() -> Path:

    RESOURCES_MODULE.mkdir(parents=True, exist_ok=True)
    init_file = RESOURCES_MODULE / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Package holding generated EDA and preprocessing diagnostic images.\n", encoding="utf-8")
    return RESOURCES_MODULE


def _save_missing_and_class_plot(rain_dataset_raw: pd.DataFrame, target_col: str, RESOURCES_MODULE: Path):

    fig, axes = plot.subplots(1, 2, figsize=(15, 5.5))

    missing_pct = rain_dataset_raw.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
    missing_pct.plot(kind="barh", ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Missing Values (%) by Feature", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Missing Percentage (%)")

    counts = rain_dataset_raw[target_col].value_counts().sort_index()
    labels = [f"No Rain ({idx})" if idx == 0 else f"Rain ({idx})" for idx in counts.index]
    axes[1].bar(labels, counts.values, color=["#3b7ea1", "#d95f02"])
    axes[1].set_title(f"Target Class Distribution ({target_col})", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Instance Count")
    for i, val in enumerate(counts.values):
        axes[1].text(i, val + 1200, f"{val:,}\n({val / len(rain_dataset_raw) * 100:.2f}%)", ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(counts.values) * 1.18)

    plot.tight_layout()
    out_path = RESOURCES_MODULE / "missing_and_class_distribution.png"
    plot.savefig(out_path, dpi=150)
    plot.close(fig)
    print(f"Saved chart -> '{out_path}'")


def _save_correlation_heatmap(rain_dataset_before: pd.DataFrame, rain_dataset_after: pd.DataFrame, RESOURCES_MODULE: Path):
    num_before = rain_dataset_before.select_dtypes(include=[np.number]).columns.tolist()
    num_after = rain_dataset_after.select_dtypes(include=[np.number]).columns.tolist()

    corr_before = rain_dataset_before[num_before].corr()
    corr_after = rain_dataset_after[num_after].corr()

    fig, axes = plot.subplots(1, 2, figsize=(20, 8.5))

    mask1 = np.triu(np.ones_like(corr_before, dtype=bool))
    sns.heatmap(
        corr_before, mask=mask1, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, linewidths=0.4, annot_kws={"size": 7.5}, ax=axes[0]
    )
    axes[0].set_title("BEFORE: All Numeric Features (Temp9am & Temp3pm Collinear)", fontsize=11, fontweight="bold")

    mask2 = np.triu(np.ones_like(corr_after, dtype=bool))
    sns.heatmap(
        corr_after, mask=mask2, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, linewidths=0.4, annot_kws={"size": 8}, ax=axes[1]
    )
    axes[1].set_title("AFTER: Retained Numeric Features (Irrelevant/Redundant Dropped)", fontsize=11, fontweight="bold")

    plot.tight_layout()
    out_path = RESOURCES_MODULE / "correlation_heatmap_before_after.png"
    plot.savefig(out_path, dpi=150)
    plot.close(fig)
    print(f"Saved chart -> '{out_path}'")


def _detect_and_plot_boxplot_outliers(
    X_train_raw: pd.DataFrame,
    num_features: list[str],
    Clipper: BoxplotOutlierClipper,
    RESOURCES_MODULE: Path,
):
    
    clean_series = [X_train_raw[col].dropna().to_numpy() for col in num_features]
    bp_stats = boxplot_stats(clean_series, labels=num_features, whis=1.5)

    print("\nTraining Set Box-Plot Outlier Detection (whisker = 1.5 * IQR):")
    for st in bp_stats:
        n_outliers = len(st["fliers"])
        if n_outliers > 0:
            print(
                f"  {st['label']:<15}: whiskers=[{st['whislo']:7.2f}, {st['whishi']:7.2f}] | "
                f"outliers detected & clipped = {n_outliers:,}"
            )

    medians = X_train_raw[num_features].median()
    iqrs = (X_train_raw[num_features].quantile(0.75) - X_train_raw[num_features].quantile(0.25)).replace(0, 1)

    rain_dataset_before_normalization = (X_train_raw[num_features] - medians) / iqrs
    clipped_vals = Clipper.transform(X_train_raw[num_features].to_numpy())
    rain_dataset_after_clipped = pd.DataFrame(clipped_vals, columns=num_features)
    rain_dataset_after_normalization = (rain_dataset_after_clipped - medians) / iqrs

    fig, axes = plot.subplots(1, 2, figsize=(18, 7), sharey=True)

    sns.boxplot(data=rain_dataset_before_normalization, orient="h", fliersize=2, color="#d95f02", ax=axes[0])
    axes[0].set_xlim(-5, 15)
    axes[0].set_title("BEFORE Outlier Adjustment: Training Set Box Plots (IQR-Scaled)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("IQR Units from Median (Fliers = Detected Outliers)")

    sns.boxplot(data=rain_dataset_after_normalization, orient="h", fliersize=2, color="#2b8cbe", ax=axes[1])
    axes[1].set_xlim(-5, 15)
    axes[1].set_title("AFTER Box-Plot Whisker Clipping: Outliers Adjusted", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("IQR Units from Median (Clipped at Whisker Bounds)")

    plot.tight_layout()
    out_path = RESOURCES_MODULE / "outliers_boxplot_before_after.png"
    plot.savefig(out_path, dpi=150)
    plot.close(fig)
    print(f"Saved chart -> '{out_path}'")


def inspect_and_clean(
    filepath: str | Path = DATA_PATH,
    target_col: str = "RainTomorrow",
    unrelated_cols: tuple[str, ...] = ("row ID", "Date", "Temp9am", "Temp3pm"),
    save_plots: bool = True,
):
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'."
        )

    RESOURCES_MODULE = _ensure_resources_module()

    if filepath.suffix.lower() in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(filepath)
        rain_dataset = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    else:
        rain_dataset = pd.read_csv(filepath)

    print(f"Loaded dataset: {rain_dataset.shape[0]} rows x {rain_dataset.shape[1]} columns")
    schema_summary = pd.DataFrame({
        "dtype": rain_dataset.dtypes.astype(str),
        "missing": rain_dataset.isnull().sum(),
        "missing_pct": (rain_dataset.isnull().mean() * 100).round(2),
        "unique_vals": rain_dataset.nunique(dropna=True),
    })
    print("\nFeature_Analysis: Initial Features, Data Types & Missing Values:")
    print(schema_summary.to_string())

    rain_dataset_before_feature_drop = rain_dataset.drop(columns=[c for c in ("row ID", "Date") if c in rain_dataset.columns]).copy()

    cols_to_drop = [col for col in unrelated_cols if col in rain_dataset.columns]
    constant_cols = [
        col for col in rain_dataset.columns
        if col != target_col and col not in cols_to_drop and rain_dataset[col].nunique(dropna=True) <= 1
    ]
    cols_to_drop.extend(constant_cols)

    if cols_to_drop:
        rain_dataset = rain_dataset.drop(columns=cols_to_drop)
        print(f"\nRemoved unrelated/redundant columns based on obtained correlation heatmap: {cols_to_drop}")

    dup_count = int(rain_dataset.duplicated().sum())
    if dup_count > 0:
        rain_dataset = rain_dataset.drop_duplicates().reset_index(drop=True)
        print(f"Removed {dup_count} duplicate rows. Clean dataset size: {len(rain_dataset)} rows")

    class_counts = rain_dataset[target_col].value_counts().sort_index()
    class_pcts = (rain_dataset[target_col].value_counts(normalize=True).sort_index() * 100).round(2)
    print("\nTarget Variable ('RainTomorrow') Class Distribution:")
    for cls_label in class_counts.index:
        print(f"  Class {cls_label}: {class_counts[cls_label]:,} instances ({class_pcts[cls_label]:.2f}%)")

    if save_plots:
        _save_missing_and_class_plot(rain_dataset_before_feature_drop, target_col, RESOURCES_MODULE)
        _save_correlation_heatmap(rain_dataset_before_feature_drop, rain_dataset, RESOURCES_MODULE)

    X = rain_dataset.drop(columns=[target_col])
    y = rain_dataset[target_col].astype(int)

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    print("Feature Engineered:")
    print(f"\nRetained Numeric Features ({len(num_features)}): {num_features}")
    print(f"Retained Categorical Features ({len(cat_features)}): {cat_features}")
    return X, y, num_features, cat_features


def preprocessor_pipeline(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_Clipper", BoxplotOutlierClipper(whisker=1.5)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ],
        remainder="drop",
    )


def split_train_test_data(
    filepath: str | Path = DATA_PATH,
    test_export_path: str | Path = DEFAULT_TEST_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
    save_plots: bool = True,
):
    
    RESOURCES_MODULE = _ensure_resources_module()
    X, y, num_features, cat_features = inspect_and_clean(filepath, save_plots=save_plots)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(
        f"\n80:20 Split -> Train: {len(X_train_raw):,} rows ({y_train.mean() * 100:.2f}% positive), "
        f"Test: {len(X_test_raw):,} rows ({y_test.mean() * 100:.2f}% positive)"
    )

   
    test_rain_dataset = X_test_raw.copy()
    test_rain_dataset["RainTomorrow"] = y_test.values
    test_rain_dataset.to_csv(test_export_path, index=False)
    print(f"Exported test split to '{test_export_path}'.")

  
    preprocessor = preprocessor_pipeline(num_features, cat_features)
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    if save_plots:
        fitted_Clipper = preprocessor.named_transformers_["num"].named_steps["outlier_Clipper"]
        _detect_and_plot_boxplot_outliers(X_train_raw, num_features, fitted_Clipper, RESOURCES_MODULE)

    feature_names = preprocessor.get_feature_names_out().tolist()
    print(f"\nProcessed feature matrix shape -> Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")

    return (
        X_train_processed,
        X_test_processed,
        y_train.to_numpy(),
        y_test.to_numpy(),
        preprocessor,
        feature_names,
    )


if __name__ == "__main__":
    split_train_test_data()