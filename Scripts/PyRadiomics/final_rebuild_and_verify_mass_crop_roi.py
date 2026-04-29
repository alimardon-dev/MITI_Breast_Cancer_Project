import os
import shutil
import pandas as pd

# =========================================================
# PATHS
# =========================================================

dataset_root = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset"

csv_root = os.path.join(dataset_root, "csv files")
raw_root = os.path.join(
    dataset_root,
    "manifest-ZkhPvrLo5216730872708713142",
    "CBIS-DDSM"
)
project_root = os.path.join(
    dataset_root,
    "manifest-ZkhPvrLo5216730872708713142",
    "project"
)
archive_root = os.path.join(project_root, "_ARCHIVE_OLD")

train_csv = os.path.join(csv_root, "mass_case_description_train_set.csv")
test_csv = os.path.join(csv_root, "mass_case_description_test_set.csv")


# =========================================================
# HELPERS
# =========================================================

def normalize_label(pathology):
    pathology = str(pathology).strip().upper()
    if pathology == "BENIGN":
        return "Benign"
    elif pathology == "MALIGNANT":
        return "Malignant"
    elif pathology == "BENIGN_WITHOUT_CALLBACK":
        return "Benign_without_callback"
    return None


def get_case_folder(path_value):
    if pd.isna(path_value):
        return None
    path_value = str(path_value).strip().replace("\\", "/")
    parts = path_value.split("/")
    if not parts:
        return None
    return parts[0]


def get_view(case_folder):
    case_folder = str(case_folder).upper()
    if "_CC" in case_folder:
        return "CC"
    elif "_MLO" in case_folder:
        return "MLO"
    return None


def get_safe_destination(base_dir, folder_name):
    candidate = os.path.join(base_dir, folder_name)
    if not os.path.exists(candidate):
        return candidate

    i = 1
    while True:
        new_candidate = os.path.join(base_dir, f"{folder_name}_{i}")
        if not os.path.exists(new_candidate):
            return new_candidate
        i += 1


def archive_folder_if_exists(folder_name):
    src = os.path.join(project_root, folder_name)
    if not os.path.exists(src):
        print(f"[SKIP ARCHIVE] {folder_name} not found")
        return

    os.makedirs(archive_root, exist_ok=True)
    dst = get_safe_destination(archive_root, folder_name)
    print(f"[ARCHIVE] {src}")
    print(f"         -> {dst}")
    shutil.move(src, dst)


def make_main_dirs(base_output):
    for view in ["CC", "MLO"]:
        for cls in ["Benign", "Malignant"]:
            os.makedirs(os.path.join(base_output, view, cls), exist_ok=True)


def find_crop_and_roi_dcms(case_folder_path):
    """
    Final confirmed rule from visual inspection:
    - smaller DCM = cropped image
    - larger DCM = ROI mask

    We search inside folders containing roi/cropped.
    """
    if not os.path.exists(case_folder_path):
        return None, None

    matches = []

    for root, dirs, files in os.walk(case_folder_path):
        root_lower = root.lower()

        if ("roi" not in root_lower) and ("cropped" not in root_lower):
            continue

        for file in files:
            if file.lower().endswith(".dcm"):
                full_path = os.path.join(root, file)
                size = os.path.getsize(full_path)
                matches.append((size, full_path))

    if len(matches) < 2:
        return None, None

    matches.sort(key=lambda x: x[0])  # ascending by size

    crop_path = matches[0][1]   # SMALLER = cropped
    roi_path = matches[-1][1]   # LARGER = ROI

    return crop_path, roi_path


def copy_file(src_path, pathology, case_folder, main_output, bwc_output):
    view = get_view(case_folder)
    if view is None:
        return "skip_unknown_view"

    src_name = os.path.basename(src_path)
    new_name = f"{case_folder}_{src_name}"

    if pathology == "Benign_without_callback":
        os.makedirs(bwc_output, exist_ok=True)
        dst_path = os.path.join(bwc_output, new_name)
    else:
        dst_path = os.path.join(main_output, view, pathology, new_name)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)

    return "copied"


def collect_case_names_from_output(base_path):
    found = set()

    if not os.path.exists(base_path):
        return found

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if not file.lower().endswith(".dcm"):
                continue

            name = os.path.splitext(file)[0]

            # expected: Mass-Training_P_00001_LEFT_CC_1_1-2
            # we want:  Mass-Training_P_00001_LEFT_CC_1
            if "_" in name:
                case_name = name.rsplit("_", 1)[0]
            else:
                case_name = name

            found.add(case_name)

    return found


def expected_case_sets_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    benign = set()
    malignant = set()
    bwc = set()

    for _, row in df.iterrows():
        pathology = normalize_label(row.get("pathology", ""))
        crop_case = get_case_folder(row.get("cropped image file path", None))
        roi_case = get_case_folder(row.get("ROI mask file path", None))

        # crop_case and roi_case should normally be same _1 folder
        case_folder = crop_case if crop_case else roi_case
        if not case_folder or pathology is None:
            continue

        # only count if raw source exists
        if not os.path.exists(os.path.join(raw_root, case_folder)):
            continue

        if pathology == "Benign":
            benign.add(case_folder)
        elif pathology == "Malignant":
            malignant.add(case_folder)
        elif pathology == "Benign_without_callback":
            bwc.add(case_folder)

    return benign, malignant, bwc


def print_check_block(title, expected_set, found_set):
    missing = sorted(expected_set - found_set)
    extra = sorted(found_set - expected_set)

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Expected cases : {len(expected_set)}")
    print(f"Found cases    : {len(found_set)}")

    if not missing and not extra:
        print("✅ PERFECT MATCH")
    else:
        if missing:
            print(f"⚠ Missing in output: {len(missing)}")
            for item in missing[:10]:
                print(" -", item)
        if extra:
            print(f"⚠ Extra in output: {len(extra)}")
            for item in extra[:10]:
                print(" -", item)


# =========================================================
# STEP 1: ARCHIVE CURRENT PARTIAL CROP/ROI OUTPUTS
# =========================================================

def archive_partial_outputs():
    print("\n" + "#" * 70)
    print("STEP 1: ARCHIVE CURRENT PARTIAL CROP/ROI OUTPUTS")
    print("#" * 70)

    folders_to_archive = [
        "Data_Mass_Train_Cropped",
        "Data_Mass_Train_ROI",
        "Data_Mass_Test_Cropped",
        "Data_Mass_Test_ROI",
        "data_mass_train_bwc_cropped",
        "data_mass_train_bwc_roi",
        "data_mass_test_bwc_cropped",
        "data_mass_test_bwc_roi",
    ]

    for folder_name in folders_to_archive:
        archive_folder_if_exists(folder_name)


# =========================================================
# STEP 2: REBUILD CROP + ROI
# =========================================================

def rebuild_split(csv_path, split_name):
    df = pd.read_csv(csv_path)

    crop_output = os.path.join(project_root, f"{split_name}_Cropped")
    roi_output = os.path.join(project_root, f"{split_name}_ROI")

    bwc_crop_output = os.path.join(project_root, f"{split_name.lower()}_bwc_cropped")
    bwc_roi_output = os.path.join(project_root, f"{split_name.lower()}_bwc_roi")

    make_main_dirs(crop_output)
    make_main_dirs(roi_output)

    stats = {
        "crop": {"copied": 0, "duplicates": 0, "missing": []},
        "roi": {"copied": 0, "duplicates": 0, "missing": []},
    }

    copied_registry = set()

    for _, row in df.iterrows():
        pathology = normalize_label(row.get("pathology", ""))
        if pathology is None:
            continue

        crop_case = get_case_folder(row.get("cropped image file path", None))
        roi_case = get_case_folder(row.get("ROI mask file path", None))

        shared_case = crop_case if crop_case else roi_case
        if not shared_case:
            continue

        case_folder_path = os.path.join(raw_root, shared_case)
        crop_dcm, roi_dcm = find_crop_and_roi_dcms(case_folder_path)

        # CROPPED
        crop_key = ("crop", pathology, shared_case)
        if crop_key in copied_registry:
            stats["crop"]["duplicates"] += 1
        else:
            if crop_dcm is None:
                stats["crop"]["missing"].append(shared_case)
            else:
                copy_file(crop_dcm, pathology, shared_case, crop_output, bwc_crop_output)
                copied_registry.add(crop_key)
                stats["crop"]["copied"] += 1

        # ROI
        roi_key = ("roi", pathology, shared_case)
        if roi_key in copied_registry:
            stats["roi"]["duplicates"] += 1
        else:
            if roi_dcm is None:
                stats["roi"]["missing"].append(shared_case)
            else:
                copy_file(roi_dcm, pathology, shared_case, roi_output, bwc_roi_output)
                copied_registry.add(roi_key)
                stats["roi"]["copied"] += 1

    print("\n" + "=" * 70)
    print(f"{split_name} REBUILD FINISHED")
    print("=" * 70)
    for kind in ["crop", "roi"]:
        print(f"{kind.upper()} copied     : {stats[kind]['copied']}")
        print(f"{kind.upper()} duplicates : {stats[kind]['duplicates']}")
        print(f"{kind.upper()} missing    : {len(set(stats[kind]['missing']))}")
        if stats[kind]["missing"]:
            print("Example missing:")
            for item in sorted(set(stats[kind]["missing"]))[:10]:
                print(" -", item)
        print("-" * 50)


# =========================================================
# STEP 3: VERIFY COUNTS
# =========================================================

def verify_outputs():
    print("\n" + "#" * 70)
    print("STEP 3: VERIFY COUNTS")
    print("#" * 70)

    train_expected_benign, train_expected_malignant, train_expected_bwc = expected_case_sets_from_csv(train_csv)
    test_expected_benign, test_expected_malignant, test_expected_bwc = expected_case_sets_from_csv(test_csv)

    # TRAIN CROPPED
    train_crop_benign_found = collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Train_Cropped", "CC", "Benign")) | \
                              collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Train_Cropped", "MLO", "Benign"))
    train_crop_malignant_found = collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Train_Cropped", "CC", "Malignant")) | \
                                 collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Train_Cropped", "MLO", "Malignant"))
    train_crop_bwc_found = collect_case_names_from_output(os.path.join(project_root, "data_mass_train_bwc_cropped"))

    # TRAIN ROI
    train_roi_benign_found = collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Train_ROI", "CC", "Benign")) | \
                             collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Train_ROI", "MLO", "Benign"))
    train_roi_malignant_found = collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Train_ROI", "CC", "Malignant")) | \
                                collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Train_ROI", "MLO", "Malignant"))
    train_roi_bwc_found = collect_case_names_from_output(os.path.join(project_root, "data_mass_train_bwc_roi"))

    # TEST CROPPED
    test_crop_benign_found = collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Test_Cropped", "CC", "Benign")) | \
                             collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Test_Cropped", "MLO", "Benign"))
    test_crop_malignant_found = collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Test_Cropped", "CC", "Malignant")) | \
                                collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Test_Cropped", "MLO", "Malignant"))
    test_crop_bwc_found = collect_case_names_from_output(os.path.join(project_root, "data_mass_test_bwc_cropped"))

    # TEST ROI
    test_roi_benign_found = collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Test_ROI", "CC", "Benign")) | \
                            collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Test_ROI", "MLO", "Benign"))
    test_roi_malignant_found = collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Test_ROI", "CC", "Malignant")) | \
                               collect_case_names_from_output(os.path.join(project_root, "Data_Mass_Test_ROI", "MLO", "Malignant"))
    test_roi_bwc_found = collect_case_names_from_output(os.path.join(project_root, "data_mass_test_bwc_roi"))

    # TRAIN CROPPED
    print_check_block("TRAIN CROPPED BENIGN CHECK", train_expected_benign, train_crop_benign_found)
    print_check_block("TRAIN CROPPED MALIGNANT CHECK", train_expected_malignant, train_crop_malignant_found)
    print_check_block("TRAIN CROPPED BWC CHECK", train_expected_bwc, train_crop_bwc_found)

    # TRAIN ROI
    print_check_block("TRAIN ROI BENIGN CHECK", train_expected_benign, train_roi_benign_found)
    print_check_block("TRAIN ROI MALIGNANT CHECK", train_expected_malignant, train_roi_malignant_found)
    print_check_block("TRAIN ROI BWC CHECK", train_expected_bwc, train_roi_bwc_found)

    # TEST CROPPED
    print_check_block("TEST CROPPED BENIGN CHECK", test_expected_benign, test_crop_benign_found)
    print_check_block("TEST CROPPED MALIGNANT CHECK", test_expected_malignant, test_crop_malignant_found)
    print_check_block("TEST CROPPED BWC CHECK", test_expected_bwc, test_crop_bwc_found)

    # TEST ROI
    print_check_block("TEST ROI BENIGN CHECK", test_expected_benign, test_roi_benign_found)
    print_check_block("TEST ROI MALIGNANT CHECK", test_expected_malignant, test_roi_malignant_found)
    print_check_block("TEST ROI BWC CHECK", test_expected_bwc, test_roi_bwc_found)


# =========================================================
# RUN ALL
# =========================================================

if __name__ == "__main__":
    archive_partial_outputs()
    rebuild_split(train_csv, "Data_Mass_Train")
    rebuild_split(test_csv, "Data_Mass_Test")
    verify_outputs()

    print("\n" + "#" * 70)
    print("ALL DONE")
    print("#" * 70)