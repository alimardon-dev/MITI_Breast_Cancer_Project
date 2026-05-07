import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "Data" / "Raw" / "CBIS-DDSM"

CSV_PATH = PROJECT_ROOT / "Data" / "Raw" / "CSV files" / "mass_case_description_test_set.csv"

OUTPUT_DIR = PROJECT_ROOT / "Data" / "CNN_Training_2.0" / "Case_Based_Benign_Malignant"

REPORT_PATH = OUTPUT_DIR / "case_based_benign_malignant_report.txt"

ABNORMALITY_SUMMARY_CSV_PATH = OUTPUT_DIR / "abnormality_counts_by_folder.csv"
ABNORMALITY_DETAILS_CSV_PATH = OUTPUT_DIR / "abnormality_details_by_folder.csv"

DATASET_PREFIX = "Mass-Test"

# In ROI mask image folders:
# 1-1.dcm = cropped image
# 1-2.dcm = ROI mask image
CROPPED_DCM_NAME = "1-1.dcm"

VALID_PATHOLOGIES = {"BENIGN", "MALIGNANT"}


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():
    for pathology in VALID_PATHOLOGIES:
        for group in ["Both_CC_MLO", "CC_Missing", "MLO_Missing"]:
            folder = OUTPUT_DIR / pathology / group
            folder.mkdir(parents=True, exist_ok=True)


def is_roi_mask_folder(path: Path):
    """
    Only process ROI mask image folders.
    In these folders:
    1-1.dcm = cropped image
    1-2.dcm = ROI mask
    """
    return "ROI mask images" in str(path)


def find_cropped_dcm_for_roi_folder(roi_folder: Path) -> Path:
    """
    Given a 'ROI mask images' folder, find the correct cropped DICOM.

    Full real structure on disk:

    Pattern #1 (1593 cases) — ROI mask folder has 2 files:
      Mass-Test_P_XXXXX_SIDE_VIEW_1/
        DD-MM-YYYY-DDSM-NA-NNNNN/
          1.000000-ROI mask images-NNNNN/
            1-1.dcm   ← cropped image  (2 files present)
            1-2.dcm   ← mask

    Pattern #3/#4 (103 cases) — two separate date folders, one per type:
      Mass-Test_P_XXXXX_SIDE_VIEW_1/
        DD-MM-YYYY-DDSM-NA-AAAAA/
          1.000000-ROI mask images-NNNNN/
            1-1.dcm   ← MASK only  (1 file = this is NOT the crop)
        DD-MM-YYYY-DDSM-NA-BBBBB/
          1.000000-cropped images-NNNNN/
            1-1.dcm   ← real CROP ✅

    Strategy:
      - Walk UP from the ROI mask folder to the _1 case root
        (roi_folder → date_folder → case_root)
      - Search the entire case_root tree for a 'cropped images' folder.
      - If one exists with 1-1.dcm, use it (Pattern #3/#4).
      - Otherwise use 1-1.dcm from the ROI mask folder itself (Pattern #1).
        In Pattern #1 the ROI mask folder has 2 files, so 1-1.dcm is the crop.
    """
    # roi_folder =  .../Mass-Test_P_XXXXX_SIDE_VIEW_1/DD-MM-YYYY-.../1.000000-ROI mask images-.../
    # date_folder = .../Mass-Test_P_XXXXX_SIDE_VIEW_1/DD-MM-YYYY-.../
    # case_root   = .../Mass-Test_P_XXXXX_SIDE_VIEW_1/
    date_folder = roi_folder.parent
    case_root = date_folder.parent

    # Search all subdirectories of the case root for a "cropped images" folder
    for sub in case_root.rglob("*"):
        if sub.is_dir() and "cropped images" in sub.name:
            crop_candidate = sub / CROPPED_DCM_NAME
            if crop_candidate.exists():
                return crop_candidate  # Pattern #3/#4

    # Fallback: Pattern #1 — 1-1.dcm inside ROI mask folder is the crop
    fallback = roi_folder / CROPPED_DCM_NAME
    if fallback.exists():
        return fallback

    return None


def parse_case_info(path: Path):
    """
    Extracts information from paths like:

    Mass-Test_P_01566_RIGHT_CC_1/.../1-1.dcm
    Mass-Test_P_01566_RIGHT_MLO_3/.../1-1.dcm
    """

    full_text = str(path)

    pattern = r"(Mass-Test_P_\d+_(LEFT|RIGHT)_(CC|MLO)(?:_(\d+))?)"
    match = re.search(pattern, full_text)

    if not match:
        return None

    case_name = match.group(1)
    side = match.group(2)
    view = match.group(3)
    abnormality_id = match.group(4)

    if abnormality_id is None:
        abnormality_id = "0"

    patient_match = re.search(r"P_\d+", case_name)
    patient_id = patient_match.group(0) if patient_match else None

    if not patient_id:
        return None

    patient_side = f"{patient_id}_{side}"
    abnormality_key = f"{patient_id}_{side}_ABN_{abnormality_id}"
    case_key = f"{patient_id}_{side}_CASE_{abnormality_id}"

    return {
        "case_name": case_name,
        "patient_id": patient_id,
        "side": side,
        "view": view,
        "abnormality_id": str(abnormality_id),
        "patient_side": patient_side,
        "abnormality_key": abnormality_key,
        "case_key": case_key,
    }


def normalize_pathology(value):
    if pd.isna(value):
        return None
    return str(value).strip().upper()


def load_csv_metadata():
    """
    Loads CSV metadata and keeps only BENIGN and MALIGNANT cases.

    Excludes:
    - BENIGN_WITHOUT_CALLBACK
    - missing/unknown pathology
    """

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    df["pathology"] = df["pathology"].apply(normalize_pathology)

    df = df[df["pathology"].isin(VALID_PATHOLOGIES)].copy()

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["left or right breast"] = df["left or right breast"].astype(str).str.strip().str.upper()
    df["image view"] = df["image view"].astype(str).str.strip().str.upper()
    df["abnormality id"] = df["abnormality id"].astype(str).str.strip()

    df["csv_abnormality_key"] = (
        df["patient_id"]
        + "_"
        + df["left or right breast"]
        + "_ABN_"
        + df["abnormality id"]
        + "_"
        + df["pathology"]
    )

    metadata = {}

    for _, row in df.iterrows():
        key = (
            row["patient_id"],
            row["left or right breast"],
            row["image view"],
            row["abnormality id"],
        )

        metadata[key] = {
            "pathology": row["pathology"],
            "abnormality_type": row.get("abnormality type", ""),
            "mass_shape": row.get("mass shape", ""),
            "mass_margins": row.get("mass margins", ""),
            "assessment": row.get("assessment", ""),
            "subtlety": row.get("subtlety", ""),
        }

    return metadata, df


def get_csv_abnormality_counts(filtered_csv):
    """
    Counts unique abnormalities from CSV.

    One abnormality =
    patient_id + side + abnormality_id + pathology
    """

    csv_abnormalities = filtered_csv.drop_duplicates(
        subset=[
            "patient_id",
            "left or right breast",
            "abnormality id",
            "pathology",
        ]
    ).copy()

    counts_by_pathology = csv_abnormalities["pathology"].value_counts().to_dict()

    total_abnormalities = len(csv_abnormalities)

    return csv_abnormalities, counts_by_pathology, total_abnormalities


def dicom_to_png(dcm_path: Path, output_png_path: Path):
    """
    Convert DICOM to PNG using min-max normalization.
    """

    dcm = pydicom.dcmread(str(dcm_path))
    img = dcm.pixel_array.astype(np.float32)

    photometric = getattr(dcm, "PhotometricInterpretation", "")

    if photometric == "MONOCHROME1":
        img = np.max(img) - img

    img = img - np.min(img)

    if np.max(img) > 0:
        img = img / np.max(img)

    img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    pil_img.save(output_png_path)


def find_cropped_dicoms(metadata):
    """
    Finds cropped DICOM files and keeps only cases that exist in CSV metadata
    with pathology BENIGN or MALIGNANT.

    Handles three patterns:

    Pattern #1 (1593 cases):
      ROI mask images/ has 2 files → 1-1.dcm is the crop, 1-2.dcm is the mask.
      → use 1-1.dcm from ROI mask images/

    Pattern #3 / #4 (103 cases):
      A sibling 'cropped images/' folder exists in the same _1 parent folder.
      → use 1-1.dcm from cropped images/ (the real crop)
      → ignore 1-1.dcm from ROI mask images/ (only 1 file = that IS the mask)
    """

    cases = []
    skipped_no_csv_match = 0
    skipped_details = []

    for root, dirs, files in os.walk(RAW_DIR):
        root_path = Path(root)

        if not is_roi_mask_folder(root_path):
            continue

        # Resolve the correct cropped DICOM for this ROI folder
        dcm_path = find_cropped_dcm_for_roi_folder(root_path)

        if dcm_path is None:
            continue

        info = parse_case_info(dcm_path)

        if info is None:
            continue

        csv_key = (
            info["patient_id"],
            info["side"],
            info["view"],
            info["abnormality_id"],
        )

        if csv_key not in metadata:
            skipped_no_csv_match += 1
            skipped_details.append({
                "dcm_path": str(dcm_path),
                "case_name": info["case_name"],
                "patient_id": info["patient_id"],
                "side": info["side"],
                "view": info["view"],
                "abnormality_id": info["abnormality_id"],
                "reason": "No BENIGN/MALIGNANT CSV match",
            })
            continue

        case_metadata = metadata[csv_key]

        cases.append({
            "dcm_path": dcm_path,
            **info,
            **case_metadata,
        })

    return cases, skipped_no_csv_match, skipped_details


def group_cases_by_abnormality_key(cases):
    """
    Groups by patient + side + abnormality_id + pathology.

    This is the abnormality-level grouping.

    Example:
    P_01566_RIGHT_CASE_1_BENIGN
    P_01566_RIGHT_CASE_2_MALIGNANT
    """

    grouped = defaultdict(list)

    for case in cases:
        group_key = f"{case['case_key']}_{case['pathology']}"
        grouped[group_key].append(case)

    return grouped


def classify_case_group(case_items):
    """
    Classify abnormality based on available views.
    """

    views = {item["view"] for item in case_items}

    if "CC" in views and "MLO" in views:
        return "Both_CC_MLO"

    if "CC" in views and "MLO" not in views:
        return "MLO_Missing"

    if "MLO" in views and "CC" not in views:
        return "CC_Missing"

    return "Unknown"


def save_case_group(group_key, case_items, group_type):
    """
    Converts cropped DICOM images to PNG and saves them into pathology folders.
    """

    pathology = case_items[0]["pathology"]

    case_output_dir = OUTPUT_DIR / pathology / group_type / group_key
    case_output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for item in case_items:
        dcm_path = item["dcm_path"]
        case_name = item["case_name"]

        output_name = f"{case_name}.png"
        output_path = case_output_dir / output_name

        dicom_to_png(dcm_path, output_path)

        saved_files.append(output_path)

    return saved_files


def count_png_files(folder: Path):
    if not folder.exists():
        return 0
    return len(list(folder.rglob("*.png")))


def count_abnormality_folders(folder: Path):
    """
    Each folder inside Both_CC_MLO / CC_Missing / MLO_Missing
    represents one abnormality group.
    """

    if not folder.exists():
        return 0

    return len([p for p in folder.iterdir() if p.is_dir()])


# ============================================================
# MAIN
# ============================================================

def main():
    print("Preparing CASE-BASED cropped Mass-Test PNG dataset...")
    print(f"Raw folder: {RAW_DIR}")
    print(f"CSV file: {CSV_PATH}")
    print(f"Output folder: {OUTPUT_DIR}")

    ensure_dirs()

    metadata, filtered_csv = load_csv_metadata()

    csv_abnormalities, csv_abn_counts_by_pathology, csv_total_abnormalities = get_csv_abnormality_counts(filtered_csv)

    print("\nCSV loaded.")
    print(f"CSV rows after keeping only BENIGN/MALIGNANT: {len(filtered_csv)}")
    print(f"CSV unique abnormalities after filtering: {csv_total_abnormalities}")
    print("CSV pathology row counts:")
    print(filtered_csv["pathology"].value_counts())
    print("CSV abnormality counts:")
    print(csv_abn_counts_by_pathology)

    all_items, skipped_no_csv_match, skipped_details = find_cropped_dicoms(metadata)

    print(f"\nFound cropped DICOM files after CSV filtering: {len(all_items)}")
    print(f"Skipped DICOM files without BENIGN/MALIGNANT CSV match: {skipped_no_csv_match}")

    grouped = group_cases_by_abnormality_key(all_items)

    print(f"Found DICOM abnormality groups: {len(grouped)}")

    stats = {
        "BENIGN": {
            "Both_CC_MLO": {"abnormalities": 0, "png_files": 0},
            "CC_Missing": {"abnormalities": 0, "png_files": 0},
            "MLO_Missing": {"abnormalities": 0, "png_files": 0},
            "Unknown": {"abnormalities": 0, "png_files": 0},
        },
        "MALIGNANT": {
            "Both_CC_MLO": {"abnormalities": 0, "png_files": 0},
            "CC_Missing": {"abnormalities": 0, "png_files": 0},
            "MLO_Missing": {"abnormalities": 0, "png_files": 0},
            "Unknown": {"abnormalities": 0, "png_files": 0},
        },
    }

    detailed_problem_cases = []
    detailed_extra_cases = []
    abnormality_details_rows = []

    for group_key, case_items in grouped.items():
        pathology = case_items[0]["pathology"]
        group_type = classify_case_group(case_items)

        views = defaultdict(list)

        for item in case_items:
            views[item["view"]].append(item["case_name"])

        cc_count = len(views["CC"])
        mlo_count = len(views["MLO"])

        if group_type == "Unknown":
            stats[pathology]["Unknown"]["abnormalities"] += 1
            stats[pathology]["Unknown"]["png_files"] += 0
            saved_files = []
        else:
            saved_files = save_case_group(group_key, case_items, group_type)

            stats[pathology][group_type]["abnormalities"] += 1
            stats[pathology][group_type]["png_files"] += len(saved_files)

        abnormality_details_rows.append({
            "folder_pathology": pathology,
            "folder_group": group_type,
            "abnormality_group_key": group_key,
            "patient_id": case_items[0]["patient_id"],
            "side": case_items[0]["side"],
            "abnormality_id": case_items[0]["abnormality_id"],
            "pathology": pathology,
            "cc_count": cc_count,
            "mlo_count": mlo_count,
            "total_png_files": len(saved_files),
            "cc_files": "; ".join(views["CC"]),
            "mlo_files": "; ".join(views["MLO"]),
        })

        if group_type in ["CC_Missing", "MLO_Missing"]:
            detailed_problem_cases.append({
                "case_key": group_key,
                "pathology": pathology,
                "problem": group_type,
                "cc_count": cc_count,
                "mlo_count": mlo_count,
                "cc_files": views["CC"],
                "mlo_files": views["MLO"],
            })

        if cc_count > 1 or mlo_count > 1:
            detailed_extra_cases.append({
                "case_key": group_key,
                "pathology": pathology,
                "cc_count": cc_count,
                "mlo_count": mlo_count,
                "cc_files": views["CC"],
                "mlo_files": views["MLO"],
            })

    # ============================================================
    # SUMMARY TABLE FOR ABNORMALITY COUNTS BY FOLDER
    # ============================================================

    summary_rows = []

    for pathology in ["BENIGN", "MALIGNANT"]:
        for group_type in ["Both_CC_MLO", "CC_Missing", "MLO_Missing", "Unknown"]:
            folder_path = OUTPUT_DIR / pathology / group_type

            summary_rows.append({
                "pathology_folder": pathology,
                "view_folder": group_type,
                "folder_path": str(folder_path),
                "abnormality_count_from_script": stats[pathology][group_type]["abnormalities"],
                "png_file_count_from_script": stats[pathology][group_type]["png_files"],
                "actual_abnormality_folders_on_disk": count_abnormality_folders(folder_path),
                "actual_png_files_on_disk": count_png_files(folder_path),
            })

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.DataFrame(abnormality_details_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(ABNORMALITY_SUMMARY_CSV_PATH, index=False)
    details_df.to_csv(ABNORMALITY_DETAILS_CSV_PATH, index=False)

    # ============================================================
    # TOTALS
    # ============================================================

    total_both_abn = (
        stats["BENIGN"]["Both_CC_MLO"]["abnormalities"]
        + stats["MALIGNANT"]["Both_CC_MLO"]["abnormalities"]
    )

    total_cc_missing_abn = (
        stats["BENIGN"]["CC_Missing"]["abnormalities"]
        + stats["MALIGNANT"]["CC_Missing"]["abnormalities"]
    )

    total_mlo_missing_abn = (
        stats["BENIGN"]["MLO_Missing"]["abnormalities"]
        + stats["MALIGNANT"]["MLO_Missing"]["abnormalities"]
    )

    total_unknown_abn = (
        stats["BENIGN"]["Unknown"]["abnormalities"]
        + stats["MALIGNANT"]["Unknown"]["abnormalities"]
    )

    total_abn_from_dicom = (
        total_both_abn
        + total_cc_missing_abn
        + total_mlo_missing_abn
        + total_unknown_abn
    )

    total_png = 0

    for pathology in ["BENIGN", "MALIGNANT"]:
        for group_type in ["Both_CC_MLO", "CC_Missing", "MLO_Missing", "Unknown"]:
            total_png += stats[pathology][group_type]["png_files"]

    # ============================================================
    # REPORT
    # ============================================================

    report_lines = []

    report_lines.append("CASE-BASED MASS TEST BENIGN/MALIGNANT ABNORMALITY DISTRIBUTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Source folder:")
    report_lines.append(str(RAW_DIR))
    report_lines.append("")
    report_lines.append("CSV file:")
    report_lines.append(str(CSV_PATH))
    report_lines.append("")
    report_lines.append("Output folder:")
    report_lines.append(str(OUTPUT_DIR))
    report_lines.append("")
    report_lines.append("Filtering rule:")
    report_lines.append("Only pathology BENIGN and MALIGNANT are included.")
    report_lines.append("BENIGN_WITHOUT_CALLBACK is excluded.")
    report_lines.append("")
    report_lines.append("Abnormality counting rule:")
    report_lines.append("One abnormality = patient_id + side + abnormality_id + pathology.")
    report_lines.append("")
    report_lines.append("Grouping rule:")
    report_lines.append("DICOM cases are grouped by patient_id + side + abnormality_id + pathology.")
    report_lines.append("")
    report_lines.append("DICOM rule:")
    report_lines.append("ROI mask image folders are used as anchors for discovery.")
    report_lines.append("Cropped image source depends on folder pattern:")
    report_lines.append("  Pattern #1 (normal): ROI mask images/ has 2 files.")
    report_lines.append("    -> 1-1.dcm = cropped image  |  1-2.dcm = mask")
    report_lines.append("    -> source: ROI mask images/1-1.dcm")
    report_lines.append("  Pattern #3/#4 (103 cases): sibling 'cropped images/' folder exists.")
    report_lines.append("    -> 1-1.dcm in 'cropped images/' is the real crop")
    report_lines.append("    -> 1-1.dcm in 'ROI mask images/' is the mask (skipped)")
    report_lines.append("    -> source: cropped images/1-1.dcm")
    report_lines.append(f"In both cases the file used is named: {CROPPED_DCM_NAME}")
    report_lines.append("")

    report_lines.append("------------------------------------------")
    report_lines.append("CSV COUNTS AFTER FILTERING")
    report_lines.append("------------------------------------------")
    report_lines.append(f"CSV image rows after BENIGN/MALIGNANT filtering: {len(filtered_csv)}")
    report_lines.append(f"CSV unique abnormalities after BENIGN/MALIGNANT filtering: {csv_total_abnormalities}")
    report_lines.append("")

    report_lines.append("CSV image row counts by pathology:")
    csv_row_counts = filtered_csv["pathology"].value_counts()
    for pathology, count in csv_row_counts.items():
        report_lines.append(f"  {pathology}: {count}")

    report_lines.append("")
    report_lines.append("CSV unique abnormality counts by pathology:")
    for pathology in ["BENIGN", "MALIGNANT"]:
        count = csv_abn_counts_by_pathology.get(pathology, 0)
        report_lines.append(f"  {pathology}: {count}")

    report_lines.append("")
    report_lines.append("------------------------------------------")
    report_lines.append("ABNORMALITY COUNTS BY OUTPUT FOLDER")
    report_lines.append("------------------------------------------")

    for pathology in ["BENIGN", "MALIGNANT"]:
        report_lines.append("")
        report_lines.append(f"{pathology}:")

        for group_type in ["Both_CC_MLO", "CC_Missing", "MLO_Missing", "Unknown"]:
            abn_count = stats[pathology][group_type]["abnormalities"]
            png_count = stats[pathology][group_type]["png_files"]

            report_lines.append(f"  {group_type}:")
            report_lines.append(f"    Abnormalities: {abn_count}")
            report_lines.append(f"    PNG files: {png_count}")

    report_lines.append("")
    report_lines.append("------------------------------------------")
    report_lines.append("TOTAL ABNORMALITY COUNTS FROM DICOM OUTPUT")
    report_lines.append("------------------------------------------")
    report_lines.append(f"Total Both_CC_MLO abnormalities: {total_both_abn}")
    report_lines.append(f"Total CC_Missing abnormalities: {total_cc_missing_abn}")
    report_lines.append(f"Total MLO_Missing abnormalities: {total_mlo_missing_abn}")
    report_lines.append(f"Total Unknown abnormalities: {total_unknown_abn}")
    report_lines.append(f"Total DICOM/output abnormalities: {total_abn_from_dicom}")
    report_lines.append(f"Total PNG files created: {total_png}")
    report_lines.append(f"Skipped DICOM files without BENIGN/MALIGNANT CSV match: {skipped_no_csv_match}")

    report_lines.append("")
    report_lines.append("------------------------------------------")
    report_lines.append("CSV VS DICOM/OUTPUT ABNORMALITY COMPARISON")
    report_lines.append("------------------------------------------")
    report_lines.append(f"CSV unique abnormalities: {csv_total_abnormalities}")
    report_lines.append(f"DICOM/output unique abnormalities: {total_abn_from_dicom}")
    report_lines.append(f"Difference CSV - DICOM/output: {csv_total_abnormalities - total_abn_from_dicom}")

    report_lines.append("")
    for pathology in ["BENIGN", "MALIGNANT"]:
        csv_count = csv_abn_counts_by_pathology.get(pathology, 0)

        output_count = (
            stats[pathology]["Both_CC_MLO"]["abnormalities"]
            + stats[pathology]["CC_Missing"]["abnormalities"]
            + stats[pathology]["MLO_Missing"]["abnormalities"]
            + stats[pathology]["Unknown"]["abnormalities"]
        )

        report_lines.append(f"{pathology}:")
        report_lines.append(f"  CSV unique abnormalities: {csv_count}")
        report_lines.append(f"  DICOM/output abnormalities: {output_count}")
        report_lines.append(f"  Difference: {csv_count - output_count}")
        report_lines.append("")

    report_lines.append("------------------------------------------")
    report_lines.append("SAVED CSV REPORTS")
    report_lines.append("------------------------------------------")
    report_lines.append(f"Abnormality summary CSV:")
    report_lines.append(str(ABNORMALITY_SUMMARY_CSV_PATH))
    report_lines.append("")
    report_lines.append(f"Abnormality details CSV:")
    report_lines.append(str(ABNORMALITY_DETAILS_CSV_PATH))
    report_lines.append("")

    report_lines.append("------------------------------------------")
    report_lines.append("PROBLEM CASES DETAILS")
    report_lines.append("------------------------------------------")

    if len(detailed_problem_cases) == 0:
        report_lines.append("No problem cases. Every included abnormality has both CC and MLO.")
    else:
        for case in detailed_problem_cases:
            report_lines.append("")
            report_lines.append(f"Case: {case['case_key']}")
            report_lines.append(f"Pathology: {case['pathology']}")
            report_lines.append(f"Problem: {case['problem']}")
            report_lines.append(f"CC files: {case['cc_count']}")
            for f in case["cc_files"]:
                report_lines.append(f"  - {f}")
            report_lines.append(f"MLO files: {case['mlo_count']}")
            for f in case["mlo_files"]:
                report_lines.append(f"  - {f}")

    report_lines.append("")
    report_lines.append("------------------------------------------")
    report_lines.append("EXTRA FILES INSIDE SAME ABNORMALITY")
    report_lines.append("------------------------------------------")

    if len(detailed_extra_cases) == 0:
        report_lines.append("No extra files inside the same abnormality.")
    else:
        for case in detailed_extra_cases:
            report_lines.append("")
            report_lines.append(f"Case: {case['case_key']}")
            report_lines.append(f"Pathology: {case['pathology']}")
            report_lines.append(f"CC files: {case['cc_count']}")
            for f in case["cc_files"]:
                report_lines.append(f"  - {f}")
            report_lines.append(f"MLO files: {case['mlo_count']}")
            for f in case["mlo_files"]:
                report_lines.append(f"  - {f}")

    report_lines.append("")
    report_lines.append("=" * 80)

    report = "\n".join(report_lines)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n✅ Report saved to: {REPORT_PATH}")
    print(f"✅ Abnormality summary CSV saved to: {ABNORMALITY_SUMMARY_CSV_PATH}")
    print(f"✅ Abnormality details CSV saved to: {ABNORMALITY_DETAILS_CSV_PATH}")
    print("✅ Done!")


if __name__ == "__main__":
    main()