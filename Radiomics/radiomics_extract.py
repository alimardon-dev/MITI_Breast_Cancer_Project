import os
import logging
import pydicom
import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor

# =========================
# 1. QUIET DOWN LOG SPAM
# =========================
logging.getLogger("radiomics").setLevel(logging.ERROR)

# =========================
# 2. PATHS AND LABELS
# =========================
base_dir = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\manifest-ZkhPvrLo5216730872708713142\project\Data"
output_dir = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\manifest-ZkhPvrLo5216730872708713142\project\Output"

os.makedirs(output_dir, exist_ok=True)

classes = {
    "Benign": 0,
    "Benign_without_callback": 0,
    "Malignant": 1
}

data = []
errors = []

# =========================
# 3. CREATE EXTRACTOR
# =========================
extractor = featureextractor.RadiomicsFeatureExtractor()

# Only use the features you want
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("firstorder")
extractor.enableFeatureClassByName("glcm")

# Important settings
extractor.settings["label"] = 1
extractor.settings["force2D"] = True
extractor.settings["distances"] = [1]
extractor.settings["binWidth"] = 16   # <-- speed fix

# =========================
# 4. LOOP THROUGH ALL FILES
# =========================
for class_name, label in classes.items():
    folder_path = os.path.join(base_dir, class_name)

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    all_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm") and "1-2" in f
    ]

    total_files = len(all_files)
    print(f"\nProcessing: {class_name} ({total_files} files)")

    for idx, file in enumerate(all_files, start=1):
        image_path = os.path.join(folder_path, file)

        try:
            # ===== Load DICOM =====
            ds = pydicom.dcmread(image_path)
            image_array = ds.pixel_array.astype(np.float32)

            # Remove singleton dimensions if present
            image_array = np.squeeze(image_array)

            # Ensure image is 2D
            if image_array.ndim != 2:
                raise ValueError(f"Expected 2D image after squeeze, got shape {image_array.shape}")

            h, w = image_array.shape

            # Skip absurdly tiny images
            if h < 3 or w < 3:
                raise ValueError(f"Image too small for mask border: shape {image_array.shape}")

            # ===== SPEED FIX: Normalize image to 0–255 =====
            img_min = image_array.min()
            img_max = image_array.max()

            if img_max > img_min:
                image_array = (image_array - img_min) / (img_max - img_min)
                image_array = image_array * 255.0
            else:
                image_array = np.zeros_like(image_array, dtype=np.float32)

            # ===== Create inner-box mask =====
            mask_array = np.zeros_like(image_array, dtype=np.uint8)

            border = 5
            if h <= 2 * border or w <= 2 * border:
                border = 1

            mask_array[border:h-border, border:w-border] = 1

            # Sanity check
            uniq = np.unique(mask_array).tolist()
            if uniq not in ([0, 1], [1]):
                raise ValueError("Mask creation failed: unexpected values")

            if np.sum(mask_array) == 0:
                raise ValueError("Mask creation failed: mask is empty")

            # ===== Convert to SimpleITK =====
            image_sitk = sitk.GetImageFromArray(image_array)
            mask_sitk = sitk.GetImageFromArray(mask_array)

            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
            mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

            # Set matching geometry
            image_sitk.SetSpacing((1.0, 1.0))
            image_sitk.SetOrigin((0.0, 0.0))
            image_sitk.SetDirection((1.0, 0.0, 0.0, 1.0))

            mask_sitk.CopyInformation(image_sitk)

            # ===== Extract features =====
            features = extractor.execute(image_sitk, mask_sitk, label=1)

            # ===== Store results =====
            row = {
                "file_name": file,
                "class_name": class_name,
                "label": label,
                "height": h,
                "width": w
            }

            for key, value in features.items():
                if "diagnostics" not in key:
                    row[key] = value

            data.append(row)

            # Progress update every 25 images
            if idx % 25 == 0 or idx == total_files:
                print(f"  {idx}/{total_files} done")

        except Exception as e:
            print(f"Error with {file}: {e}")
            errors.append({
                "file_name": file,
                "class_name": class_name,
                "error": str(e)
            })

# =========================
# 5. SAVE RESULTS
# =========================
df = pd.DataFrame(data)
errors_df = pd.DataFrame(errors)

features_path = os.path.join(output_dir, "features.csv")
errors_path = os.path.join(output_dir, "feature_extraction_errors.csv")

df.to_csv(features_path, index=False)
errors_df.to_csv(errors_path, index=False)

print("\nDONE ✅")
print(f"Features saved to: {features_path}")
print(f"Errors saved to:   {errors_path}")
print(f"Total successful samples: {len(df)}")
print(f"Total failed samples: {len(errors_df)}")