import os
import pandas as pd

# =========================================================
# PATHS
# =========================================================

dataset_root = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset"

raw_cbis_root = os.path.join(
    dataset_root,
    "manifest-ZkhPvrLo5216730872708713142",
    "CBIS-DDSM"
)

project_root = os.path.join(
    dataset_root,
    "manifest-ZkhPvrLo5216730872708713142",
    "project"
)

csv_root = os.path.join(dataset_root, "csv files")

train_csv = os.path.join(csv_root, "mass_case_description_train_set.csv")
test_csv = os.path.join(csv_root, "mass_case_description_test_set.csv")

processed_train_root = os.path.join(project_root, "Data_Mass_Train")
processed_test_root = os.path.join(project_root, "Data_Mass_Test")

train_bwc_root = os.path.join(project_root, "train_bwc")
test_bwc_root = os.path.join(project_root, "test_bwc")


# =========================================================
# HELPERS
# =========================================================

def get_case_folder_from_image_path(image_file_path):
    """
    Example:
    Mass-Training_P_00016_LEFT_CC/1.3.6.1.../000000.dcm
    -> Mass-Training_P_00016_LEFT_CC
    """
    if pd.isna(image_file_path):
        return None
    image_file_path = str(image_file_path).strip().replace("\\", "/")
    parts = image_file_path.split("/")
    if not parts:
        return None
    return parts[0]


def collect_expected_cases_from_csv(csv_path):
    """
    Returns sets of case folder names for:
    benign, malignant, bwc
    """
    df = pd.read_csv(csv_path)

    benign_cases = set()
    malignant_cases = set()
    bwc_cases = set()

    for _, row in df.iterrows():
        pathology = str(row.get("pathology", "")).strip().upper()
        image_path = row.get("image file path", None)
        case_folder = get_case_folder_from_image_path(image_path)

        if not case_folder:
            continue

        if pathology == "BENIGN":
            benign_cases.add(case_folder)
        elif pathology == "MALIGNANT":
            malignant_cases.add(case_folder)
        elif pathology == "BENIGN_WITHOUT_CALLBACK":
            bwc_cases.add(case_folder)

    return benign_cases, malignant_cases, bwc_cases


def collect_processed_case_names_from_class_structure(base_path):
    """
    For structure like:
    Data_Mass_Train/
        CC/Benign/
        CC/Malignant/
        MLO/Benign/
        MLO/Malignant/

    Extract case names from filenames like:
    Mass-Training_P_00016_LEFT_CC_1-1.dcm
    -> case name = Mass-Training_P_00016_LEFT_CC
    """
    found_cases = set()

    if not os.path.exists(base_path):
        return found_cases

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if not file.lower().endswith(".dcm"):
                continue

            # remove extension
            name = os.path.splitext(file)[0]

            # expected processed filename format:
            # Mass-Training_P_00016_LEFT_CC_1-1
            # We want everything except the last "_1-1"
            if "_" in name:
                case_name = name.rsplit("_", 1)[0]
            else:
                case_name = name

            found_cases.add(case_name)

    return found_cases


def collect_processed_case_names_from_bwc_folder(base_path):
    """
    For train_bwc / test_bwc where files are like:
    Mass-Training_P_00016_LEFT_CC_1-1.dcm
    """
    found_cases = set()

    if not os.path.exists(base_path):
        return found_cases

    for file in os.listdir(base_path):
        file_path = os.path.join(base_path, file)

        if os.path.isdir(file_path):
            continue
        if not file.lower().endswith(".dcm"):
            continue

        name = os.path.splitext(file)[0]
        if "_" in name:
            case_name = name.rsplit("_", 1)[0]
        else:
            case_name = name

        found_cases.add(case_name)

    return found_cases


def check_original_existence(case_set, raw_root):
    """
    Checks which expected case folders actually exist in the raw CBIS-DDSM folder.
    """
    existing = set()
    missing = set()

    for case in case_set:
        case_path = os.path.join(raw_root, case)
        if os.path.exists(case_path):
            existing.add(case)
        else:
            missing.add(case)

    return existing, missing


def print_check_block(title, expected_set, found_set):
    missing_in_processed = sorted(expected_set - found_set)
    extra_in_processed = sorted(found_set - expected_set)

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Expected cases : {len(expected_set)}")
    print(f"Found cases    : {len(found_set)}")

    if not missing_in_processed and not extra_in_processed:
        print("✅ PERFECT MATCH")
    else:
        if missing_in_processed:
            print(f"⚠ Missing in processed: {len(missing_in_processed)}")
            print("   Example(s):")
            for item in missing_in_processed[:10]:
                print("   -", item)

        if extra_in_processed:
            print(f"⚠ Extra in processed: {len(extra_in_processed)}")
            print("   Example(s):")
            for item in extra_in_processed[:10]:
                print("   -", item)


# =========================================================
# MAIN CHECK
# =========================================================

# ---- TRAIN expected from CSV
train_benign_expected, train_malignant_expected, train_bwc_expected = collect_expected_cases_from_csv(train_csv)

# ---- TEST expected from CSV
test_benign_expected, test_malignant_expected, test_bwc_expected = collect_expected_cases_from_csv(test_csv)

# ---- Check whether expected CSV-listed raw folders really exist
train_benign_existing, train_benign_missing_raw = check_original_existence(train_benign_expected, raw_cbis_root)
train_malignant_existing, train_malignant_missing_raw = check_original_existence(train_malignant_expected, raw_cbis_root)
train_bwc_existing, train_bwc_missing_raw = check_original_existence(train_bwc_expected, raw_cbis_root)

test_benign_existing, test_benign_missing_raw = check_original_existence(test_benign_expected, raw_cbis_root)
test_malignant_existing, test_malignant_missing_raw = check_original_existence(test_malignant_expected, raw_cbis_root)
test_bwc_existing, test_bwc_missing_raw = check_original_existence(test_bwc_expected, raw_cbis_root)

# ---- Collect processed train cases
train_cc_benign_found = collect_processed_case_names_from_class_structure(
    os.path.join(processed_train_root, "CC", "Benign")
)
train_mlo_benign_found = collect_processed_case_names_from_class_structure(
    os.path.join(processed_train_root, "MLO", "Benign")
)
train_benign_found = train_cc_benign_found | train_mlo_benign_found

train_cc_malignant_found = collect_processed_case_names_from_class_structure(
    os.path.join(processed_train_root, "CC", "Malignant")
)
train_mlo_malignant_found = collect_processed_case_names_from_class_structure(
    os.path.join(processed_train_root, "MLO", "Malignant")
)
train_malignant_found = train_cc_malignant_found | train_mlo_malignant_found

train_bwc_found = collect_processed_case_names_from_bwc_folder(train_bwc_root)

# ---- Collect processed test cases
test_cc_benign_found = collect_processed_case_names_from_class_structure(
    os.path.join(processed_test_root, "CC", "Benign")
)
test_mlo_benign_found = collect_processed_case_names_from_class_structure(
    os.path.join(processed_test_root, "MLO", "Benign")
)
test_benign_found = test_cc_benign_found | test_mlo_benign_found

test_cc_malignant_found = collect_processed_case_names_from_class_structure(
    os.path.join(processed_test_root, "CC", "Malignant")
)
test_mlo_malignant_found = collect_processed_case_names_from_class_structure(
    os.path.join(processed_test_root, "MLO", "Malignant")
)
test_malignant_found = test_cc_malignant_found | test_mlo_malignant_found

test_bwc_found = collect_processed_case_names_from_bwc_folder(test_bwc_root)

# =========================================================
# REPORT
# =========================================================

print("\n" + "#" * 70)
print("VERIFY MASS TRAIN / TEST STRUCTURE")
print("#" * 70)

# --- Raw existence summary
print("\nRAW DATASET EXISTENCE CHECK")
print("-" * 70)
print(f"Train BENIGN existing raw folders                : {len(train_benign_existing)}")
print(f"Train MALIGNANT existing raw folders             : {len(train_malignant_existing)}")
print(f"Train BENIGN_WITHOUT_CALLBACK existing raw       : {len(train_bwc_existing)}")
print(f"Test BENIGN existing raw folders                 : {len(test_benign_existing)}")
print(f"Test MALIGNANT existing raw folders              : {len(test_malignant_existing)}")
print(f"Test BENIGN_WITHOUT_CALLBACK existing raw        : {len(test_bwc_existing)}")

if train_benign_missing_raw:
    print(f"\n⚠ Train BENIGN missing from raw dataset: {len(train_benign_missing_raw)}")
    for item in sorted(train_benign_missing_raw)[:10]:
        print(" -", item)

if train_malignant_missing_raw:
    print(f"\n⚠ Train MALIGNANT missing from raw dataset: {len(train_malignant_missing_raw)}")
    for item in sorted(train_malignant_missing_raw)[:10]:
        print(" -", item)

if train_bwc_missing_raw:
    print(f"\n⚠ Train BWC missing from raw dataset: {len(train_bwc_missing_raw)}")
    for item in sorted(train_bwc_missing_raw)[:10]:
        print(" -", item)

if test_benign_missing_raw:
    print(f"\n⚠ Test BENIGN missing from raw dataset: {len(test_benign_missing_raw)}")
    for item in sorted(test_benign_missing_raw)[:10]:
        print(" -", item)

if test_malignant_missing_raw:
    print(f"\n⚠ Test MALIGNANT missing from raw dataset: {len(test_malignant_missing_raw)}")
    for item in sorted(test_malignant_missing_raw)[:10]:
        print(" -", item)

if test_bwc_missing_raw:
    print(f"\n⚠ Test BWC missing from raw dataset: {len(test_bwc_missing_raw)}")
    for item in sorted(test_bwc_missing_raw)[:10]:
        print(" -", item)

# --- Main checks: only benign + malignant matter most
print_check_block(
    "TRAIN BENIGN CHECK",
    train_benign_existing,
    train_benign_found
)

print_check_block(
    "TRAIN MALIGNANT CHECK",
    train_malignant_existing,
    train_malignant_found
)

print_check_block(
    "TEST BENIGN CHECK",
    test_benign_existing,
    test_benign_found
)

print_check_block(
    "TEST MALIGNANT CHECK",
    test_malignant_existing,
    test_malignant_found
)

# --- BWC separate report
print_check_block(
    "TRAIN BWC CHECK (SEPARATE REPORT)",
    train_bwc_existing,
    train_bwc_found
)

print_check_block(
    "TEST BWC CHECK (SEPARATE REPORT)",
    test_bwc_existing,
    test_bwc_found
)

print("\n" + "#" * 70)
print("DONE")
print("#" * 70)