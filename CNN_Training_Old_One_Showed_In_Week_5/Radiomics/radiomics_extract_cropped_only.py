import os
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize

warnings.filterwarnings("ignore")

# =========================
# PATHS
# =========================
BASE_DIR = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\manifest-ZkhPvrLo5216730872708713142\project"

DATA_DIR = os.path.join(BASE_DIR, "Data_Mass_Train_Cropped")
SAVE_DIR = os.path.join(BASE_DIR, "extracted_features_cropped_only", "Mass_train")

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================
TARGET_SIZE = (256, 256)

LABEL_MAP = {
    "Benign": 0,
    "Malignant": 1
}

META_COLS = ["label", "class_name", "view", "image_name"]

# =========================
# PYRADIOMICS EXTRACTOR
# =========================
settings = {
    "force2D": True,
    "force2Ddimension": 0,
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": "sitkBSpline",
    "verbose": False,
    "label": 1
}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

extractor.disableAllFeatures()
extractor.enableFeatureClassByName("firstorder")
extractor.enableFeatureClassByName("shape2D")
extractor.enableFeatureClassByName("glcm")
extractor.enableFeatureClassByName("glrlm")
extractor.enableFeatureClassByName("glszm")
extractor.enableFeatureClassByName("gldm")
extractor.enableFeatureClassByName("ngtdm")

extractor.disableAllImageTypes()
extractor.enableImageTypeByName("Original")

# =========================
# HELPERS
# =========================
def load_and_resize_dcm(path, target_size=(256, 256)):
    image = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(image)

    # convert to 2D
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected image shape after squeeze: {arr.shape}")

    # resize image for speed
    resized_arr = resize(
        arr,
        target_size,
        order=1,               # bilinear for image
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)

    resized_img = sitk.GetImageFromArray(resized_arr)

    # full mask = whole image
    mask_arr = np.ones(target_size, dtype=np.uint8)
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(resized_img)

    return resized_img, mask_img


def clean_features(result_dict):
    return {
        k: v for k, v in result_dict.items()
        if not k.startswith("diagnostics")
    }


def remove_shape(df):
    return df[[c for c in df.columns if "shape" not in c.lower()]]


def normalize(df):
    feature_cols = [c for c in df.columns if c not in META_COLS]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])

    df_scaled = pd.DataFrame(scaled, columns=feature_cols)

    for col in META_COLS:
        df_scaled[col] = df[col].values

    return df_scaled[META_COLS + feature_cols]


def process_view(view):
    rows = []

    for class_name in ["Benign", "Malignant"]:
        folder = os.path.join(DATA_DIR, view, class_name)

        if not os.path.exists(folder):
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]

        print(f"\n[INFO] {view} - {class_name}")
        print(f"Total cases: {len(files)}")

        for f in tqdm(files):
            try:
                path = os.path.join(folder, f)

                image, mask = load_and_resize_dcm(path, TARGET_SIZE)

                result = extractor.execute(image, mask, label=1)
                features = clean_features(result)

                features["label"] = LABEL_MAP[class_name]
                features["class_name"] = class_name
                features["view"] = view
                features["image_name"] = f

                rows.append(features)

            except Exception as e:
                print(f"[ERROR] {f}: {e}")

    return pd.DataFrame(rows)

# =========================
# RUN
# =========================
print("START FAST CROPPED-ONLY EXTRACTION")

cc_df = process_view("CC")
mlo_df = process_view("MLO")

# raw
cc_raw_path = os.path.join(SAVE_DIR, "cc_extracted_features.csv")
mlo_raw_path = os.path.join(SAVE_DIR, "mlo_extracted_features.csv")

cc_df.to_csv(cc_raw_path, index=False)
mlo_df.to_csv(mlo_raw_path, index=False)

print("\nSaved RAW CSVs")

# without shape
cc_no_shape = remove_shape(cc_df)
mlo_no_shape = remove_shape(mlo_df)

cc_no_shape_path = os.path.join(SAVE_DIR, "cc_extracted_features_without_shape.csv")
mlo_no_shape_path = os.path.join(SAVE_DIR, "mlo_extracted_features_without_shape.csv")

cc_no_shape.to_csv(cc_no_shape_path, index=False)
mlo_no_shape.to_csv(mlo_no_shape_path, index=False)

print("Saved WITHOUT SHAPE CSVs")

# normalized
cc_norm = normalize(cc_df)
mlo_norm = normalize(mlo_df)
cc_no_shape_norm = normalize(cc_no_shape)
mlo_no_shape_norm = normalize(mlo_no_shape)

cc_norm.to_csv(
    os.path.join(SAVE_DIR, "cc_extracted_features_normalized.csv"),
    index=False
)
mlo_norm.to_csv(
    os.path.join(SAVE_DIR, "mlo_extracted_features_normalized.csv"),
    index=False
)
cc_no_shape_norm.to_csv(
    os.path.join(SAVE_DIR, "cc_extracted_features_without_shape_normalized.csv"),
    index=False
)
mlo_no_shape_norm.to_csv(
    os.path.join(SAVE_DIR, "mlo_extracted_features_without_shape_normalized.csv"),
    index=False
)

print("\nALL DONE")
print(f"Saved folder: {SAVE_DIR}")