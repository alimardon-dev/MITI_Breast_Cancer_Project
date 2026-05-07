import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RADIOMICS_DIR = PROJECT_ROOT / "RESULTS" / "Radiomics_Filtered_For_Fusion"

# 🔥 IMPORTANT: you must have these from late fusion
CNN_PROBS_DIR = PROJECT_ROOT / "RESULTS" / "LateFusion"

OUTPUT_DIR = PROJECT_ROOT / "RESULTS" / "Hybrid_Model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_FEATURES = 30   # try 20 / 30 / 40

CLASS_NAMES = ["Benign", "Malignant"]


# =========================
# LOAD DATA
# =========================

def load_data(split):
    rad_path = RADIOMICS_DIR / f"{split}_radiomics_for_fusion.csv"
    cnn_path = CNN_PROBS_DIR / f"{split}_cnn_probs.csv"

    rad_df = pd.read_csv(rad_path)
    cnn_df = pd.read_csv(cnn_path)

    merged = rad_df.merge(cnn_df, on="patient_id")

    return merged


# =========================
# PREPARE FEATURES
# =========================

def prepare_features(df):
    y = df["label"].values

    # CNN features
    cnn_features = df[["cnn_prob_benign", "cnn_prob_malignant"]].values

    # Radiomics features (everything starting with cc_ or mlo_)
    rad_cols = [col for col in df.columns if col.startswith("cc_") or col.startswith("mlo_")]
    rad_features = df[rad_cols].values

    return cnn_features, rad_features, y, rad_cols


# =========================
# MAIN
# =========================

def main():
    print("Loading data...")

    train_df = load_data("train")
    val_df = load_data("val")
    test_df = load_data("test")

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Prepare features
    X_cnn_train, X_rad_train, y_train, rad_cols = prepare_features(train_df)
    X_cnn_val, X_rad_val, y_val, _ = prepare_features(val_df)
    X_cnn_test, X_rad_test, y_test, _ = prepare_features(test_df)

    # =========================
    # FEATURE SELECTION
    # =========================

    print(f"\nSelecting top {TOP_K_FEATURES} radiomics features...")

    selector = SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)

    X_rad_train_selected = selector.fit_transform(X_rad_train, y_train)
    X_rad_val_selected = selector.transform(X_rad_val)
    X_rad_test_selected = selector.transform(X_rad_test)

    selected_features = [rad_cols[i] for i in selector.get_support(indices=True)]

    print("Selected features:")
    for f in selected_features[:10]:
        print(f"  {f}")
    print("...")

    # =========================
    # COMBINE CNN + RADIOMICS
    # =========================

    X_train = np.hstack([X_cnn_train, X_rad_train_selected])
    X_val = np.hstack([X_cnn_val, X_rad_val_selected])
    X_test = np.hstack([X_cnn_test, X_rad_test_selected])

    # =========================
    # TRAIN MODEL
    # =========================

    print("\nTraining hybrid model...")

    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train, y_train)

    # =========================
    # EVALUATION
    # =========================

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    print("\nVALIDATION RESULTS")
    print(classification_report(y_val, val_preds, target_names=CLASS_NAMES))

    print("\nTEST RESULTS")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES))

    # =========================
    # SAVE RESULTS
    # =========================

    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w") as f:
        f.write("Hybrid Model (CNN + Radiomics)\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Top K features: {TOP_K_FEATURES}\n\n")

        f.write("VALIDATION RESULTS\n")
        f.write(classification_report(y_val, val_preds, target_names=CLASS_NAMES))

        f.write("\nTEST RESULTS\n")
        f.write(classification_report(y_test, test_preds, target_names=CLASS_NAMES))

    print(f"\nSaved report: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")

    plt.title("Hybrid Model Confusion Matrix")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    print("Saved confusion matrix")


if __name__ == "__main__":
    main()