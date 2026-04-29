import os
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm

# =========================
# PATHS (KEEP SAME)
# =========================
train_csv = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\csv files\mass_case_description_train_set.csv"

dataset_root = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM"

output_dir = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\CBIS_DDSM_TEST"

# Create output folders
for cls in ["b", "m"]:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)


# =========================
# LABEL FUNCTION
# =========================
def get_label(pathology):
    pathology = str(pathology).strip().upper()
    if pathology in ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
        return "b"
    elif pathology == "MALIGNANT":
        return "m"
    return None


# =========================
# LOAD CSV
# =========================
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    df = df[["pathology", "cropped image file path"]].copy()
    df = df.dropna(subset=["cropped image file path"])

    df["label"] = df["pathology"].apply(get_label)
    df = df.dropna(subset=["label"])

    return df


# =========================
# EXTRACT FOLDER NAME
# =========================
def extract_case_folder(path):
    path = path.replace("\\", "/")
    return path.split("/")[0]


# =========================
# FIND DICOM
# =========================
def find_dcm(case_folder):
    folder_path = os.path.join(dataset_root, case_folder)

    if not os.path.isdir(folder_path):
        print(f"❌ Missing folder: {case_folder}")
        return None

    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".dcm"):
                return os.path.join(root, f)

    print(f"❌ No DICOM in: {case_folder}")
    return None


# =========================
# SAVE IMAGE
# =========================
def save_dicom_as_png(dicom_path, save_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype("uint8")

    cv2.imwrite(save_path, img)


# =========================
# PROCESS ONLY 5
# =========================
def process_small(df):
    df = df.head(5)

    success = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = row["label"]
        path = row["cropped image file path"]

        case_folder = extract_case_folder(path)
        dicom_path = find_dcm(case_folder)

        if dicom_path is None:
            continue

        filename = case_folder + ".png"
        save_path = os.path.join(output_dir, label, filename)

        try:
            save_dicom_as_png(dicom_path, save_path)
            success += 1
        except Exception as e:
            print(f"⚠️ Error: {e}")

    print(f"\n✅ Saved {success}/5 images")


# =========================
# RUN TEST
# =========================
if __name__ == "__main__":
    df = load_and_prepare(train_csv)
    process_small(df)