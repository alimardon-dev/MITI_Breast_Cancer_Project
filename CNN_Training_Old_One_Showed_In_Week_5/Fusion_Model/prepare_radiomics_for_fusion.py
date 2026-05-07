import re
from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CNN_BOTH_DIR = PROJECT_ROOT / "Data" / "CNN_Training" / "CNN_Both"

RADIOMICS_DIR = PROJECT_ROOT / "Radiomics" / "Fusion_Features"

TRAIN_CC_CSV = RADIOMICS_DIR / "train_cc_radiomics_without_shape_normalized.csv"
TRAIN_MLO_CSV = RADIOMICS_DIR / "train_mlo_radiomics_without_shape_normalized.csv"

TEST_CC_CSV = RADIOMICS_DIR / "test_cc_radiomics_without_shape_normalized.csv"
TEST_MLO_CSV = RADIOMICS_DIR / "test_mlo_radiomics_without_shape_normalized.csv"

OUTPUT_DIR = PROJECT_ROOT / "RESULTS" / "Radiomics_Filtered_For_Fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Benign", "Malignant"]
SPLITS = ["train", "val", "test"]


# =========================
# HELPERS
# =========================

def extract_patient_id(text):
    match = re.search(r"P_\d+", str(text))
    return match.group(0) if match else None


def get_cnn_both_patients(split):
    rows = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = CNN_BOTH_DIR / split / class_name

        if not class_dir.exists():
            print(f"Missing folder: {class_dir}")
            continue

        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            cc_path = patient_dir / "CC.png"
            mlo_path = patient_dir / "MLO.png"

            if cc_path.exists() and mlo_path.exists():
                rows.append({
                    "patient_id": patient_dir.name,
                    "label": label,
                    "class_name": class_name,
                    "split": split
                })

    return pd.DataFrame(rows)


def load_radiomics_file(csv_path, view_prefix):
    df = pd.read_csv(csv_path)

    if "patient_id" not in df.columns:
        if "image_name" not in df.columns:
            raise ValueError(f"{csv_path} needs either patient_id or image_name column.")
        df["patient_id"] = df["image_name"].apply(extract_patient_id)

    df = df.dropna(subset=["patient_id"])

    meta_cols = [
        "split",
        "patient_id",
        "view",
        "label",
        "class_name",
        "image_name"
    ]

    feature_cols = [
        col for col in df.columns
        if col not in meta_cols
    ]

    # Force numeric only
    usable_cols = []
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().sum() > 0:
            usable_cols.append(col)

    df = df[["patient_id"] + usable_cols]

    df = df.replace([np.inf, -np.inf], np.nan)

    # If same patient appears multiple times, average the features
    df = df.groupby("patient_id", as_index=False).mean()

    # Prefix features to avoid CC/MLO name collision
    rename_map = {
        col: f"{view_prefix}_{col}"
        for col in df.columns
        if col != "patient_id"
    }

    df = df.rename(columns=rename_map)

    print(f"Loaded {view_prefix.upper()} radiomics from {csv_path.name}")
    print(f"  Patients: {len(df)}")
    print(f"  Features: {len(df.columns) - 1}")

    return df


def load_all_radiomics():
    cc_train = load_radiomics_file(TRAIN_CC_CSV, "cc")
    cc_test = load_radiomics_file(TEST_CC_CSV, "cc")

    mlo_train = load_radiomics_file(TRAIN_MLO_CSV, "mlo")
    mlo_test = load_radiomics_file(TEST_MLO_CSV, "mlo")

    cc_all = pd.concat([cc_train, cc_test], ignore_index=True)
    mlo_all = pd.concat([mlo_train, mlo_test], ignore_index=True)

    # If duplicate patients exist after concat, average them
    cc_feature_cols = [c for c in cc_all.columns if c != "patient_id"]
    mlo_feature_cols = [c for c in mlo_all.columns if c != "patient_id"]

    cc_all = cc_all.groupby("patient_id", as_index=False)[cc_feature_cols].mean()
    mlo_all = mlo_all.groupby("patient_id", as_index=False)[mlo_feature_cols].mean()

    print("\nCombined radiomics:")
    print(f"  CC patients: {len(cc_all)}")
    print(f"  MLO patients: {len(mlo_all)}")

    return cc_all, mlo_all


def build_split_file(split, cc_all, mlo_all):
    cnn_df = get_cnn_both_patients(split)

    merged = cnn_df.merge(cc_all, on="patient_id", how="inner")
    merged = merged.merge(mlo_all, on="patient_id", how="inner")

    output_path = OUTPUT_DIR / f"{split}_radiomics_for_fusion.csv"
    merged.to_csv(output_path, index=False)

    benign_count = (merged["label"] == 0).sum()
    malignant_count = (merged["label"] == 1).sum()

    return {
        "split": split,
        "cnn_patients": len(cnn_df),
        "merged_patients": len(merged),
        "skipped": len(cnn_df) - len(merged),
        "benign": benign_count,
        "malignant": malignant_count,
        "path": output_path
    }


def save_report(results):
    report_lines = []

    report_lines.append("Radiomics Filtering Report for Fusion")
    report_lines.append("=" * 45)
    report_lines.append("")
    report_lines.append("Input radiomics:")
    report_lines.append(f"Train CC: {TRAIN_CC_CSV}")
    report_lines.append(f"Train MLO: {TRAIN_MLO_CSV}")
    report_lines.append(f"Test CC: {TEST_CC_CSV}")
    report_lines.append(f"Test MLO: {TEST_MLO_CSV}")
    report_lines.append("")
    report_lines.append("Rule:")
    report_lines.append("Only patients found in CNN_Both split AND CC radiomics AND MLO radiomics are kept.")
    report_lines.append("")

    for r in results:
        report_lines.append(r["split"].upper())
        report_lines.append("-" * 20)
        report_lines.append(f"CNN_Both patients: {r['cnn_patients']}")
        report_lines.append(f"After CC + MLO radiomics merge: {r['merged_patients']}")
        report_lines.append(f"Skipped patients: {r['skipped']}")
        report_lines.append(f"Benign: {r['benign']}")
        report_lines.append(f"Malignant: {r['malignant']}")
        report_lines.append(f"Saved: {r['path']}")
        report_lines.append("")

    report_path = OUTPUT_DIR / "filtering_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ Report saved: {report_path}")


# =========================
# MAIN
# =========================

def main():
    print("Preparing radiomics for fusion...\n")

    cc_all, mlo_all = load_all_radiomics()

    results = []

    for split in SPLITS:
        result = build_split_file(split, cc_all, mlo_all)
        results.append(result)

        print(f"\n{split.upper()}")
        print("-" * 20)
        print(f"CNN_Both patients: {result['cnn_patients']}")
        print(f"Merged patients: {result['merged_patients']}")
        print(f"Skipped: {result['skipped']}")
        print(f"Benign: {result['benign']}")
        print(f"Malignant: {result['malignant']}")
        print(f"Saved: {result['path']}")

    save_report(results)

    print("\n✅ Radiomics fusion preparation finished!")
    print(f"📁 Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()