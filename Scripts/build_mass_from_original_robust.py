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
    else:
        return None


def get_case_folder(path_value):
    """
    Example:
    Mass-Training_P_00001_LEFT_CC/.../000000.dcm
    ->
    Mass-Training_P_00001_LEFT_CC
    """
    if pd.isna(path_value):
        return None

    path_value = str(path_value).strip().replace("\\", "/")
    parts = path_value.split("/")

    if len(parts) == 0:
        return None

    return parts[0]


def get_view(case_folder):
    case_folder = str(case_folder).upper()

    if "_CC" in case_folder:
        return "CC"
    elif "_MLO" in case_folder:
        return "MLO"
    else:
        return None


def make_output_dirs(base_output):
    """
    Create:
    base_output/
        CC/Benign
        CC/Malignant
        MLO/Benign
        MLO/Malignant
    """
    for view in ["CC", "MLO"]:
        for cls in ["Benign", "Malignant"]:
            os.makedirs(os.path.join(base_output, view, cls), exist_ok=True)


def find_dcm_by_kind(case_folder_path, kind):
    """
    kind can be:
      - 'full'
      - 'crop'
      - 'roi'

    Search inside a case folder by internal folder names:
      full -> folder path contains "full mammogram images"
      crop -> folder path contains "cropped images"
      roi  -> folder path contains "ROI mask images"
    """
    if not os.path.exists(case_folder_path):
        return None

    matches = []

    for root, dirs, files in os.walk(case_folder_path):
        root_lower = root.lower()

        for file in files:
            if not file.lower().endswith(".dcm"):
                continue

            if kind == "full":
                if "full mammogram images" in root_lower:
                    matches.append(os.path.join(root, file))

            elif kind == "crop":
                if "cropped images" in root_lower:
                    matches.append(os.path.join(root, file))

            elif kind == "roi":
                if "roi mask images" in root_lower:
                    matches.append(os.path.join(root, file))

    # Prefer specific standard filenames if present
    if kind in ["full", "crop"]:
        for path in matches:
            if os.path.basename(path).lower() == "000000.dcm":
                return path

    if kind == "roi":
        for path in matches:
            if os.path.basename(path).lower() == "000001.dcm":
                return path

    # fallback
    if matches:
        matches.sort()
        return matches[0]

    return None


def copy_case(case_folder, pathology, kind, main_output, bwc_output, copied_registry):
    """
    Copy one case into the proper destination.
    """
    view = get_view(case_folder)
    if view is None:
        return "skip_unknown_view"

    case_folder_path = os.path.join(raw_root, case_folder)
    src_dcm = find_dcm_by_kind(case_folder_path, kind)

    if src_dcm is None:
        return f"missing::{case_folder}"

    unique_key = (case_folder, pathology, kind)
    if unique_key in copied_registry:
        return "duplicate_skip"

    copied_registry.add(unique_key)

    src_name = os.path.basename(src_dcm)
    new_name = f"{case_folder}_{src_name}"

    if pathology == "Benign_without_callback":
        os.makedirs(bwc_output, exist_ok=True)
        dst_path = os.path.join(bwc_output, new_name)
    else:
        dst_path = os.path.join(main_output, view, pathology, new_name)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_dcm, dst_path)

    return "copied"


def process_split(csv_path, split_name):
    """
    Build:
      split_name_Full
      split_name_Cropped
      split_name_ROI

    and separate:
      split_name.lower()_bwc_full
      split_name.lower()_bwc_cropped
      split_name.lower()_bwc_roi
    """
    df = pd.read_csv(csv_path)

    full_output = os.path.join(project_root, f"{split_name}_Full")
    crop_output = os.path.join(project_root, f"{split_name}_Cropped")
    roi_output = os.path.join(project_root, f"{split_name}_ROI")

    bwc_full_output = os.path.join(project_root, f"{split_name.lower()}_bwc_full")
    bwc_crop_output = os.path.join(project_root, f"{split_name.lower()}_bwc_cropped")
    bwc_roi_output = os.path.join(project_root, f"{split_name.lower()}_bwc_roi")

    make_output_dirs(full_output)
    make_output_dirs(crop_output)
    make_output_dirs(roi_output)

    copied_registry = set()

    stats = {
        "full": {"copied": 0, "missing": [], "duplicates": 0},
        "crop": {"copied": 0, "missing": [], "duplicates": 0},
        "roi": {"copied": 0, "missing": [], "duplicates": 0},
    }

    for _, row in df.iterrows():
        pathology = normalize_label(row.get("pathology", ""))
        if pathology is None:
            continue

        full_case = get_case_folder(row.get("image file path", None))
        crop_case = get_case_folder(row.get("cropped image file path", None))
        roi_case = get_case_folder(row.get("ROI mask file path", None))

        # FULL
        if full_case:
            result = copy_case(
                case_folder=full_case,
                pathology=pathology,
                kind="full",
                main_output=full_output,
                bwc_output=bwc_full_output,
                copied_registry=copied_registry
            )

            if result == "copied":
                stats["full"]["copied"] += 1
            elif result == "duplicate_skip":
                stats["full"]["duplicates"] += 1
            elif result.startswith("missing::"):
                stats["full"]["missing"].append(result.split("::", 1)[1])

        # CROPPED
        if crop_case:
            result = copy_case(
                case_folder=crop_case,
                pathology=pathology,
                kind="crop",
                main_output=crop_output,
                bwc_output=bwc_crop_output,
                copied_registry=copied_registry
            )

            if result == "copied":
                stats["crop"]["copied"] += 1
            elif result == "duplicate_skip":
                stats["crop"]["duplicates"] += 1
            elif result.startswith("missing::"):
                stats["crop"]["missing"].append(result.split("::", 1)[1])

        # ROI
        if roi_case:
            result = copy_case(
                case_folder=roi_case,
                pathology=pathology,
                kind="roi",
                main_output=roi_output,
                bwc_output=bwc_roi_output,
                copied_registry=copied_registry
            )

            if result == "copied":
                stats["roi"]["copied"] += 1
            elif result == "duplicate_skip":
                stats["roi"]["duplicates"] += 1
            elif result.startswith("missing::"):
                stats["roi"]["missing"].append(result.split("::", 1)[1])

    print("\n" + "=" * 70)
    print(f"{split_name} FINISHED")
    print("=" * 70)

    for kind in ["full", "crop", "roi"]:
        print(f"{kind.upper()} copied     : {stats[kind]['copied']}")
        print(f"{kind.upper()} duplicates : {stats[kind]['duplicates']}")
        print(f"{kind.upper()} missing    : {len(set(stats[kind]['missing']))}")

        if stats[kind]["missing"]:
            print("Example missing:")
            for item in sorted(set(stats[kind]["missing"]))[:10]:
                print(" -", item)

        print("-" * 50)


# =========================================================
# RUN
# =========================================================

process_split(train_csv, "Data_Mass_Train")
process_split(test_csv, "Data_Mass_Test")

print("\nALL DONE.")