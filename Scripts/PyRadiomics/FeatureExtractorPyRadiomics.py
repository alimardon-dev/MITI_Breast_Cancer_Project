import os
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

# =========================
# FILE PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

roi_path = os.path.join(BASE_DIR, "MalignantSamplePNG", "Mass-Training_P_00001_LEFT_CC_1_1-1.dcm")
image_path = os.path.join(BASE_DIR, "MalignantSamplePNG", "Mass-Training_P_00001_LEFT_CC_1_1-2.dcm")

print("Image path:", image_path)
print("ROI path:", roi_path)

# =========================
# READ IMAGE + MASK
# =========================
image = sitk.ReadImage(image_path)   # cropped image
mask = sitk.ReadImage(roi_path)      # ROI mask

print("\n--- ORIGINAL INFO ---")
print("Image size:", image.GetSize())
print("Mask size:", mask.GetSize())
print("Image spacing:", image.GetSpacing())
print("Mask spacing:", mask.GetSpacing())

# =========================
# CHECK MASK VALUES
# =========================
mask_arr = sitk.GetArrayFromImage(mask)
unique_vals, counts = np.unique(mask_arr, return_counts=True)

print("\n--- MASK UNIQUE VALUES ---")
for v, c in zip(unique_vals[:20], counts[:20]):
    print(f"Value: {v}, Count: {c}")

# =========================
# BINARIZE ROI MASK
# =========================
max_val = int(mask_arr.max())

if max_val == 0:
    raise ValueError("Mask is empty.")

binary_mask = sitk.BinaryThreshold(
    mask,
    lowerThreshold=1,
    upperThreshold=max_val,
    insideValue=1,
    outsideValue=0
)

binary_arr = sitk.GetArrayFromImage(binary_mask)
lesion_pixels = int(np.sum(binary_arr == 1))

print("\nLesion pixels:", lesion_pixels)

if lesion_pixels <= 1:
    raise ValueError("Mask has too few lesion pixels.")

# =========================
# EXTRACT FEATURES
# =========================
extractor = featureextractor.RadiomicsFeatureExtractor()

print("\nRunning PyRadiomics...")
result = extractor.execute(image, binary_mask, label=1)

features = {k: v for k, v in result.items() if not k.startswith("diagnostics")}

print("\n--- FEATURES ---")
print("Total extracted:", len(features))

for i, (k, v) in enumerate(features.items()):
    print(f"{k}: {v}")
    if i >= 19:
        break