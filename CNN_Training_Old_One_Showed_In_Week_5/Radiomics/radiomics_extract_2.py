import os
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from skimage.transform import resize

# =========================================================
# PATHS
# =========================================================

project_root = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\manifest-ZkhPvrLo5216730872708713142\project"

cropped_root = os.path.join(project_root, "Data_Mass_Train_Cropped")
roi_root = os.path.join(project_root, "Data_Mass_Train_ROI")

output_root = os.path.join(project_root, "extracted_features_cropped", "Mass_train")
os.makedirs(output_root, exist_ok=True)

cc_output_csv = os.path.join(output_root, "cc_extracted_features.csv")
mlo_output_csv = os.path.join(output_root, "mlo_extracted_features.csv")

# =========================================================
# SETTINGS
# =========================================================

LABEL_MAP = {
    "Benign": 0,
    "Malignant": 1
}

VIEWS = ["CC", "MLO"]
CLASSES = ["Benign", "Malignant"]

# =========================================================
# PYRADIOMICS SETUP
# =========================================================

extractor = featureextractor.RadiomicsFeatureExtractor()

extractor.disableAllFeatures()

# Intensity
extractor.enableFeatureClassByName("firstorder")

# Shape
extractor.enableFeatureClassByName("shape2D")

# Texture
extractor.enableFeatureClassByName("glcm")
extractor.enableFeatureClassByName("glrlm")
extractor.enableFeatureClassByName("glszm")
extractor.enableFeatureClassByName("gldm")
extractor.enableFeatureClassByName("ngtdm")

extractor.disableAllImageTypes()
extractor.enableImageTypeByName("Original")

extractor.settings["force2D"] = True
extractor.settings["force2Ddimension"] = 0
extractor.settings["label"] = 1

# =========================================================
# HELPERS
# =========================================================

def normalize_case_id(filename: str) -> str:
    """
    Example:
    Mass-Training_P_00001_LEFT_CC_1_1-2.dcm
    ->
    Mass-Training_P_00001_LEFT_CC_1
    """
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_\d+-\d+$", "", name)
    return name


def collect_files(folder_path: str) -> dict:
    """
    Returns:
      { case_id: full_path }
    """
    files_map = {}

    if not os.path.exists(folder_path):
        return files_map

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".dcm"):
            continue

        case_id = normalize_case_id(file)
        files_map[case_id] = os.path.join(folder_path, file)

    return files_map


def read_image(path):
    return sitk.ReadImage(path)


def prepare_mask(mask_path, image):
    """
    Read ROI mask, binarize it, resize by array shape to match image.
    This avoids spatial-coordinate mismatch issues.
    """
    mask = sitk.ReadImage(mask_path)

    # Convert both to arrays
    mask_arr = sitk.GetArrayFromImage(mask)
    img_arr = sitk.GetArrayFromImage(image)

    if mask_arr.max() == 0:
        raise ValueError("Mask is empty")

    # Binary mask: any non-zero -> 1
    mask_arr = (mask_arr > 0).astype(np.uint8)

    # Resize mask to image shape
    resized_mask_arr = resize(
        mask_arr,
        img_arr.shape,
        order=0,               # nearest neighbor
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)

    lesion_pixels = int(np.sum(resized_mask_arr == 1))

    if lesion_pixels <= 10:
        raise ValueError("Mask too small after resize")

    # Convert back to SimpleITK image and copy image metadata
    resized_mask = sitk.GetImageFromArray(resized_mask_arr)
    resized_mask.CopyInformation(image)

    return resized_mask, lesion_pixels


def clean_features(result):
    """
    Remove diagnostics fields.
    """
    return {k: v for k, v in result.items() if not k.startswith("diagnostics")}


# =========================================================
# MAIN EXTRACTION
# =========================================================

def extract_view(view):
    rows = []

    for class_name in CLASSES:
        label = LABEL_MAP[class_name]

        cropped_folder = os.path.join(cropped_root, view, class_name)
        roi_folder = os.path.join(roi_root, view, class_name)

        cropped_files = collect_files(cropped_folder)
        roi_files = collect_files(roi_folder)

        common_case_ids = sorted(set(cropped_files.keys()) & set(roi_files.keys()))

        print(f"\n[INFO] {view} - {class_name}")
        print(f"Cropped files: {len(cropped_files)}")
        print(f"ROI files    : {len(roi_files)}")
        print(f"Matched pairs: {len(common_case_ids)}")

        success = 0
        fail = 0

        for i, case_id in enumerate(common_case_ids, 1):
            image_path = cropped_files[case_id]
            mask_path = roi_files[case_id]

            try:
                image = read_image(image_path)
                mask, lesion_pixels = prepare_mask(mask_path, image)

                result = extractor.execute(image, mask, label=1)
                features = clean_features(result)

                size = image.GetSize()

                row = {
                    "case_id": case_id,
                    "file_name": os.path.basename(image_path),
                    "label": label,
                    "class_name": class_name,
                    "view": view,
                    "width": int(size[0]) if len(size) > 0 else None,
                    "height": int(size[1]) if len(size) > 1 else None,
                    "lesion_pixels": lesion_pixels,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }

                row.update(features)
                rows.append(row)
                success += 1

                if i % 50 == 0:
                    print(f"Processed {i}/{len(common_case_ids)}")

            except Exception as e:
                fail += 1
                print(f"[ERROR] {case_id} -> {e}")

        print(f"SUCCESS: {success}")
        print(f"FAILED : {fail}")

    return pd.DataFrame(rows)


# =========================================================
# RUN
# =========================================================

print("=" * 60)
print("START EXTRACTION")
print("=" * 60)

cc_df = extract_view("CC")
cc_df.to_csv(cc_output_csv, index=False)

mlo_df = extract_view("MLO")
mlo_df.to_csv(mlo_output_csv, index=False)

print("\nDONE")
print(f"CC rows : {len(cc_df)}")
print(f"MLO rows: {len(mlo_df)}")
print(f"Saved: {cc_output_csv}")
print(f"Saved: {mlo_output_csv}")