import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADIOMICS_DIR = PROJECT_ROOT / "RESULTS" / "Radiomics_Filtered_For_Fusion"

CLASS_NAMES = ["Benign", "Malignant"]

K_VALUES = [3, 5, 8, 10, 15, 20]


# =========================
# LOAD
# =========================

def load_split(split):
    path = RADIOMICS_DIR / f"{split}_radiomics_for_fusion.csv"
    df = pd.read_csv(path)

    # Remove duplicates safely
    label_df = df[["patient_id", "label"]].drop_duplicates("patient_id")

    df_numeric = df.drop(columns=["label", "class_name", "split"], errors="ignore")
    df_numeric = df_numeric.groupby("patient_id", as_index=False).mean(numeric_only=True)

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

    return X.values, y


# =========================
# MAIN
# =========================

def main():
    print("Loading radiomics...")

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train, y_train = get_features(train_df)
    X_val, y_val = get_features(val_df)
    X_test, y_test = get_features(test_df)

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    best_result = None

    for k in K_VALUES:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        k_actual = min(k, X_train.shape[1])

        selector = SelectKBest(f_classif, k=k_actual)
        X_train_sel = selector.fit_transform(X_train_s, y_train)
        X_val_sel = selector.transform(X_val_s)
        X_test_sel = selector.transform(X_test_s)

        model = LogisticRegression(max_iter=5000, class_weight="balanced")
        model.fit(X_train_sel, y_train)

        val_preds = model.predict(X_val_sel)
        val_acc = accuracy_score(y_val, val_preds)

        print(f"K={k_actual} → Val Acc={val_acc:.4f}")

        if best_result is None or val_acc > best_result["val_acc"]:
            best_result = {
                "k": k_actual,
                "model": model,
                "selector": selector,
                "scaler": scaler,
                "val_acc": val_acc
            }

    print("\nBest K:", best_result["k"])

    # TEST
    X_test_s = best_result["scaler"].transform(X_test)
    X_test_sel = best_result["selector"].transform(X_test_s)

    test_preds = best_result["model"].predict(X_test_sel)

    print("\nTEST REPORT")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()