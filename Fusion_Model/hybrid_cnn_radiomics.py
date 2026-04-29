import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, f1_score


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CNN_PROBS_DIR = PROJECT_ROOT / "RESULTS" / "CNN_Probs_For_Hybrid"
RADIOMICS_DIR = PROJECT_ROOT / "RESULTS" / "Radiomics_Filtered_For_Fusion"

OUTPUT_DIR = PROJECT_ROOT / "RESULTS" / "Hybrid_CNN_Radiomics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Benign", "Malignant"]

# Try different K values, choose best using validation
K_VALUES = [10, 20, 30, 40, 50, 70, 100]

SEED = 42


# =========================
# LOAD + MERGE
# =========================

def load_split(split):
    cnn_path = CNN_PROBS_DIR / f"{split}_cnn_probs.csv"
    rad_path = RADIOMICS_DIR / f"{split}_radiomics_for_fusion.csv"

    cnn_df = pd.read_csv(cnn_path)
    rad_df = pd.read_csv(rad_path)

    print(f"\n{split.upper()} duplicate check:")
    print("CNN duplicate patients:", cnn_df["patient_id"].duplicated().sum())
    print("Radiomics duplicate patients:", rad_df["patient_id"].duplicated().sum())

    # Keep labels from CNN file before averaging numeric probs
    label_df = cnn_df[["patient_id", "label"]].drop_duplicates("patient_id")

    # Average duplicate CNN probability rows
    cnn_numeric = cnn_df.drop(columns=["label"], errors="ignore")
    cnn_numeric = cnn_numeric.groupby("patient_id", as_index=False).mean(numeric_only=True)

    cnn_df = label_df.merge(cnn_numeric, on="patient_id", how="inner")

    # Remove duplicate radiomics rows
    rad_df = rad_df.drop(columns=["label", "class_name", "split"], errors="ignore")
    rad_df = rad_df.groupby("patient_id", as_index=False).mean(numeric_only=True)

    merged = cnn_df.merge(rad_df, on="patient_id", how="inner")

    print(f"{split.upper()}: CNN={len(cnn_df)}, Radiomics={len(rad_df)}, Merged={len(merged)}")

    return merged

def get_features(df):
    y = df["label"].values

    cnn_cols = [
        "cnn_prob_benign",
        "cnn_prob_malignant"
    ]

    rad_cols = [
        col for col in df.columns
        if col.startswith("cc_") or col.startswith("mlo_")
    ]

    feature_cols = cnn_cols + rad_cols

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X.values, y, feature_cols


# =========================
# TRAIN + EVALUATE
# =========================

def train_with_k(X_train, y_train, X_val, y_val, k):
    # Remove constant features
    variance_filter = VarianceThreshold(threshold=0.0)
    X_train_v = variance_filter.fit_transform(X_train)
    X_val_v = variance_filter.transform(X_val)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_v)
    X_val_s = scaler.transform(X_val_v)

    # Select top K
    actual_k = min(k, X_train_s.shape[1])
    selector = SelectKBest(score_func=f_classif, k=actual_k)
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_val_sel = selector.transform(X_val_s)

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=SEED
    )

    model.fit(X_train_sel, y_train)

    val_preds = model.predict(X_val_sel)

    acc = accuracy_score(y_val, val_preds)
    mal_recall = recall_score(y_val, val_preds, pos_label=1)
    macro_f1 = f1_score(y_val, val_preds, average="macro")

    return {
        "k": actual_k,
        "model": model,
        "variance_filter": variance_filter,
        "scaler": scaler,
        "selector": selector,
        "val_acc": acc,
        "val_mal_recall": mal_recall,
        "val_macro_f1": macro_f1
    }


def transform_with_pipeline(X, pipeline):
    X_v = pipeline["variance_filter"].transform(X)
    X_s = pipeline["scaler"].transform(X_v)
    X_sel = pipeline["selector"].transform(X_s)
    return X_sel


def main():
    print("Loading data...")

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train, y_train, feature_cols = get_features(train_df)
    X_val, y_val, _ = get_features(val_df)
    X_test, y_test, _ = get_features(test_df)

    print("\nFeature count before selection:", X_train.shape[1])
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    # =========================
    # Tune K on validation
    # =========================

    results = []

    print("\nTuning K using validation set...")

    for k in K_VALUES:
        result = train_with_k(X_train, y_train, X_val, y_val, k)
        results.append(result)

        print(
            f"K={result['k']} | "
            f"Val Acc={result['val_acc']:.4f} | "
            f"Val Mal Recall={result['val_mal_recall']:.4f} | "
            f"Val Macro F1={result['val_macro_f1']:.4f}"
        )

    # Choose best:
    # 1st priority: macro F1
    # 2nd priority: malignant recall
    # 3rd priority: accuracy
    best = sorted(
        results,
        key=lambda r: (r["val_macro_f1"], r["val_mal_recall"], r["val_acc"]),
        reverse=True
    )[0]

    print("\nBest K:", best["k"])

    # =========================
    # Final evaluation
    # =========================

    X_val_final = transform_with_pipeline(X_val, best)
    X_test_final = transform_with_pipeline(X_test, best)

    val_preds = best["model"].predict(X_val_final)
    test_preds = best["model"].predict(X_test_final)

    val_report = classification_report(
        y_val,
        val_preds,
        target_names=CLASS_NAMES
    )

    test_report = classification_report(
        y_test,
        test_preds,
        target_names=CLASS_NAMES
    )

    print("\nVALIDATION REPORT")
    print(val_report)

    print("\nTEST REPORT")
    print(test_report)

    # =========================
    # Save selected feature names
    # =========================

    kept_after_variance = np.array(feature_cols)[best["variance_filter"].get_support()]
    selected_features = kept_after_variance[best["selector"].get_support()]

    selected_path = OUTPUT_DIR / "selected_features.txt"

    with open(selected_path, "w", encoding="utf-8") as f:
        f.write(f"Best K: {best['k']}\n\n")
        for feat in selected_features:
            f.write(str(feat) + "\n")

    # =========================
    # Save report
    # =========================

    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Hybrid Fusion Model: CNN Late Fusion 0.5/0.5 + Radiomics\n")
        f.write("=" * 65 + "\n\n")

        f.write("Data:\n")
        f.write(f"Train: {len(train_df)} patients\n")
        f.write(f"Val: {len(val_df)} patients\n")
        f.write(f"Test: {len(test_df)} patients\n\n")

        f.write(f"Initial feature count: {X_train.shape[1]}\n")
        f.write(f"Best K selected on validation: {best['k']}\n\n")

        f.write("Validation tuning results:\n")
        for r in results:
            f.write(
                f"K={r['k']} | "
                f"Val Acc={r['val_acc']:.4f} | "
                f"Val Mal Recall={r['val_mal_recall']:.4f} | "
                f"Val Macro F1={r['val_macro_f1']:.4f}\n"
            )

        f.write("\nVALIDATION REPORT\n")
        f.write(val_report)

        f.write("\nTEST REPORT\n")
        f.write(test_report)

    # =========================
    # Save confusion matrix
    # =========================

    cm = confusion_matrix(y_test, test_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(cmap="Blues")
    plt.title("Hybrid CNN + Radiomics Confusion Matrix")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save predictions
    pred_df = test_df[["patient_id", "label"]].copy()
    pred_df["predicted_label"] = test_preds
    pred_df["true_label_name"] = [CLASS_NAMES[i] for i in y_test]
    pred_df["predicted_label_name"] = [CLASS_NAMES[i] for i in test_preds]
    pred_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print("\n✅ Hybrid model finished!")
    print("📁 Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()