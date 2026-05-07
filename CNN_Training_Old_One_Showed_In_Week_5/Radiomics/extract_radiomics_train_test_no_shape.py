import os
import re
import logging
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

from pathlib import Path
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "Data" / "Cropped" / "Data_Mass_Train_Cropped"
TEST_DIR = PROJECT_ROOT / "Data" / "Cropped" / "Data_Mass_Test_Cropped"

OUTPUT_DIR = PROJECT_ROOT / "Radiomics" / "Fusion_Features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = ["CC", "MLO"]
CLASSES = ["Benign", "Malignant"]

META_COLS = ["split", "patient_id", "view", "label", "class_name", "image_name"]


# Reduce repeated PyRadiomics messages
logging.getLogger("radiomics").setLevel(logging.ERROR)


# =========================
# HELPERS
# =========================

def extract_patient_id(filename):
    match = re.search(r"P_\d+", str(filename))
    return match.group(0) if match else None


def read_dicom_as_sitk(dcm_path):
    ds = pydicom.dcmread(str(dcm_path))
    img = ds.pixel_array.astype(np.float32)

    # Normalize raw pixel values to 0–1 before PyRadiomics
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    sitk_img = sitk.GetImageFromArray(img)

    # Mask = non-zero cropped region
    mask = (img > 0).astype(np.uint8)

    # Safety: PyRadiomics needs at least some mask pixels
    if mask.sum() == 0:
        return None, None

    sitk_mask = sitk.GetImageFromArray(mask)

    return sitk_img, sitk_mask


def build_extractor():
    settings = {
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100,
        "removeOutliers": 3,
        "force2D": True,
        "force2Ddimension": 0,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    extractor.disableAllFeatures()

    # Intensity + texture features only
    # Shape is intentionally excluded because our mask is generated from crop > 0.
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")

    return extractor


def extract_from_folder(base_dir, split_name, extractor):
    rows = []
    error_count = 0
    skipped_count = 0

    print(f"\nExtracting {split_name.upper()} from:")
    print(base_dir)

    for view in VIEWS:
        for class_name in CLASSES:
            class_dir = base_dir / view / class_name

            if not class_dir.exists():
                print(f"Missing folder: {class_dir}")
                continue

            label = 0 if class_name == "Benign" else 1
            dcm_files = list(class_dir.glob("*.dcm"))

            print(f"{split_name} | {view} | {class_name}: {len(dcm_files)} DICOM files")

            for dcm_file in dcm_files:
                patient_id = extract_patient_id(dcm_file.name)

                if patient_id is None:
                    skipped_count += 1
                    print(f"Skipping no patient ID: {dcm_file.name}")
                    continue

                try:
                    image, mask = read_dicom_as_sitk(dcm_file)

                    if image is None or mask is None:
                        skipped_count += 1
                        print(f"Skipping empty mask: {dcm_file.name}")
                        continue

                    result = extractor.execute(image, mask)

                    row = {
                        "split": split_name,
                        "patient_id": patient_id,
                        "view": view,
                        "label": label,
                        "class_name": class_name,
                        "image_name": dcm_file.name,
                    }

                    for key, value in result.items():
                        key = str(key)

                        if key.startswith("diagnostics"):
                            continue

                        if "shape" in key.lower():
                            continue

                        # Keep only scalar numeric values
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            row[key] = float(value)

                        elif isinstance(value, np.ndarray):
                            if value.size == 1:
                                row[key] = float(value.item())
                            else:
                                continue

                        else:
                            try:
                                row[key] = float(value)
                            except:
                                continue

                    rows.append(row)

                except Exception as e:
                    error_count += 1
                    print(f"Error processing {dcm_file.name}: {e}")

    df = pd.DataFrame(rows)

    print(f"\n{split_name.upper()} extraction finished")
    print(f"Rows extracted: {len(df)}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")

    return df


def clean_numeric_features(train_df, test_df):
    possible_feature_cols = [
        col for col in train_df.columns
        if col not in META_COLS
    ]

    print("\nPossible feature columns:", len(possible_feature_cols))

    # Convert PyRadiomics outputs safely to numeric
    for col in possible_feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
        else:
            test_df[col] = np.nan

    # Keep only columns with at least one valid value in train
    feature_cols = [
        col for col in possible_feature_cols
        if train_df[col].notna().sum() > 0
    ]

    print("Usable numeric feature columns:", len(feature_cols))

    if len(feature_cols) == 0:
        print("\nDEBUG:")
        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)
        print("Train columns:", train_df.columns.tolist()[:30])
        raise ValueError("No usable numeric radiomics features found.")

    return train_df, test_df, feature_cols


def normalize_train_test(train_df, test_df):
    if train_df.empty:
        raise ValueError("Train dataframe is empty. Check folder paths or extraction errors.")

    if test_df.empty:
        print("WARNING: Test dataframe is empty. Continuing, but test CSV will be empty.")

    train_df, test_df, feature_cols = clean_numeric_features(train_df, test_df)

    train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Fill missing values using TRAIN medians only
    medians = train_df[feature_cols].median()

    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(medians)

    # Final fallback if a column median was NaN
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    scaler = StandardScaler()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

    if not test_df.empty:
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    print("Normalization complete using TRAIN only.")

    return train_df, test_df, feature_cols


def save_view_csvs(df, split_name):
    for view in VIEWS:
        view_df = df[df["view"] == view].copy()

        output_path = OUTPUT_DIR / f"{split_name}_{view.lower()}_radiomics_without_shape_normalized.csv"
        view_df.to_csv(output_path, index=False)

        print(f"Saved: {output_path} | rows: {len(view_df)}")


def save_report(train_df, test_df, feature_cols):
    report_path = OUTPUT_DIR / "radiomics_extraction_report.txt"

    lines = []
    lines.append("Radiomics Extraction Report")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Settings:")
    lines.append("- Features: firstorder, glcm, glrlm, glszm, gldm, ngtdm")
    lines.append("- Shape features: excluded")
    lines.append("- Normalization: StandardScaler fitted on TRAIN only")
    lines.append("")
    lines.append(f"Train rows: {len(train_df)}")
    lines.append(f"Test rows: {len(test_df)}")
    lines.append(f"Usable numeric features: {len(feature_cols)}")
    lines.append("")

    for split_name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        lines.append(split_name)
        lines.append("-" * 20)

        if df.empty:
            lines.append("No rows")
            lines.append("")
            continue

        for view in VIEWS:
            for class_name in CLASSES:
                count = len(df[(df["view"] == view) & (df["class_name"] == class_name)])
                lines.append(f"{view} {class_name}: {count}")

        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved report: {report_path}")


# =========================
# MAIN
# =========================

def main():
    extractor = build_extractor()

    train_df = extract_from_folder(TRAIN_DIR, "train", extractor)
    test_df = extract_from_folder(TEST_DIR, "test", extractor)

    print("\nTrain dataframe shape:", train_df.shape)
    print("Test dataframe shape:", test_df.shape)

    train_df, test_df, feature_cols = normalize_train_test(train_df, test_df)

    train_path = OUTPUT_DIR / "train_radiomics_without_shape_normalized.csv"
    test_path = OUTPUT_DIR / "test_radiomics_without_shape_normalized.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved: {train_path}")
    print(f"Saved: {test_path}")

    save_view_csvs(train_df, "train")
    save_view_csvs(test_df, "test")

    save_report(train_df, test_df, feature_cols)

    print("\nDone ✅")


if __name__ == "__main__":
    main()