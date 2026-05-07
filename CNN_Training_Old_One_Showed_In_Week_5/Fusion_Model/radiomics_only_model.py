import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RADIOMICS_DIR = PROJECT_ROOT / "RESULTS" / "Radiomics_Filtered_For_Fusion"

OUTPUT_DIR = PROJECT_ROOT / "RESULTS" / "Radiomics_Only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Benign", "Malignant"]

K_VALUES = [3, 5, 8, 10, 15, 20]


# =========================
# LOAD DATA
# =========================

def load_split(split):
    path = RADIOMICS_DIR / f"{split}_radiomics_for_fusion.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    # Remove duplicates safely
    label_df = df[["patient_id", "label"]].drop_duplicates("patient_id")

    df_numeric = df.drop(
        columns=["label", "class_name", "split"],
        errors="ignore"
    )

    df_numeric = df_numeric.groupby(
        "patient_id",
        as_index=False
    ).mean(numeric_only=True)

    df = label_df.merge(df_numeric, on="patient_id", how="inner")

    return df


def get_features(df):
    y = df["label"].values

    feature_cols = [
        col for col in df.columns
        if col.startswith("cc_") or col.startswith("mlo_")
    ]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X.values, y, feature_cols


# =========================
# MAIN
# =========================

def main():
    print("Loading radiomics...")

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train, y_train, feature_cols = get_features(train_df)
    X_val, y_val, _ = get_features(val_df)
    X_test, y_test, _ = get_features(test_df)

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    best_result = None
    tuning_results = []

    for k in K_VALUES:
        scaler = StandardScaler()

        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        k_actual = min(k, X_train.shape[1])

        selector = SelectKBest(score_func=f_classif, k=k_actual)

        X_train_sel = selector.fit_transform(X_train_s, y_train)
        X_val_sel = selector.transform(X_val_s)
        X_test_sel = selector.transform(X_test_s)

        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced"
        )

        model.fit(X_train_sel, y_train)

        val_preds = model.predict(X_val_sel)
        val_acc = accuracy_score(y_val, val_preds)

        print(f"K={k_actual} → Val Acc={val_acc:.4f}")

        tuning_results.append({
            "k": k_actual,
            "val_acc": val_acc
        })

        if best_result is None or val_acc > best_result["val_acc"]:
            best_result = {
                "k": k_actual,
                "model": model,
                "selector": selector,
                "scaler": scaler,
                "val_acc": val_acc,
                "X_test_sel": X_test_sel
            }

    print("\nBest K:", best_result["k"])

    # =========================
    # TEST EVALUATION
    # =========================

    X_test_s = best_result["scaler"].transform(X_test)
    X_test_sel = best_result["selector"].transform(X_test_s)

    test_preds = best_result["model"].predict(X_test_sel)

    test_report = classification_report(
        y_test,
        test_preds,
        target_names=CLASS_NAMES
    )

    print("\nTEST REPORT")
    print(test_report)

    # =========================
    # SAVE CLASSIFICATION REPORT
    # =========================

    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Radiomics-Only Model Results\n")
        f.write("============================\n\n")

        f.write("Data:\n")
        f.write(f"Train patients: {len(train_df)}\n")
        f.write(f"Validation patients: {len(val_df)}\n")
        f.write(f"Test patients: {len(test_df)}\n\n")

        f.write(f"Initial radiomics feature count: {X_train.shape[1]}\n")
        f.write(f"Best K selected on validation: {best_result['k']}\n")
        f.write(f"Best validation accuracy: {best_result['val_acc']:.4f}\n\n")

        f.write("Validation tuning results:\n")
        for r in tuning_results:
            f.write(f"K={r['k']} | Val Accuracy={r['val_acc']:.4f}\n")

        f.write("\nTEST REPORT\n")
        f.write(test_report)

    print(f"Saved classification report: {report_path}")

    # =========================
    # SAVE CONFUSION MATRIX
    # =========================

    cm = confusion_matrix(y_test, test_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(cmap="Blues")
    plt.title("Radiomics-Only Confusion Matrix")
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")

    # =========================
    # SAVE SELECTED FEATURES
    # =========================

    selected_indices = best_result["selector"].get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]

    selected_path = OUTPUT_DIR / "selected_features.txt"

    with open(selected_path, "w", encoding="utf-8") as f:
        f.write(f"Best K: {best_result['k']}\n\n")
        for feature in selected_features:
            f.write(feature + "\n")

    print(f"Saved selected features: {selected_path}")

    print("\n✅ Radiomics-only model finished!")
    print(f"📁 Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()