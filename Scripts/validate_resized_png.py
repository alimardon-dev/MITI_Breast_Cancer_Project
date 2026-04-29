import os
from collections import defaultdict
from PIL import Image

ORIG_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\Data_Mass_Train_Cropped_PNG"
RESIZED_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\Data_Mass_Train_Cropped_PNG_RESIZED"
TARGET_SIZE = (224, 224)

counts_orig = defaultdict(int)
counts_resized = defaultdict(int)
sizes_found = set()
errors = []

for view in ["CC", "MLO"]:
    for label in ["Benign", "Malignant"]:
        orig_folder = os.path.join(ORIG_ROOT, view, label)
        resized_folder = os.path.join(RESIZED_ROOT, view, label)

        if not os.path.exists(orig_folder):
            errors.append(f"Missing original folder: {orig_folder}")
            continue

        if not os.path.exists(resized_folder):
            errors.append(f"Missing resized folder: {resized_folder}")
            continue

        orig_files = sorted([f for f in os.listdir(orig_folder) if f.lower().endswith(".png")])
        resized_files = sorted([f for f in os.listdir(resized_folder) if f.lower().endswith(".png")])

        counts_orig[(view, label)] = len(orig_files)
        counts_resized[(view, label)] = len(resized_files)

        orig_set = set(orig_files)
        resized_set = set(resized_files)

        missing_after_resize = orig_set - resized_set
        extra_after_resize = resized_set - orig_set

        if missing_after_resize:
            errors.append(f"{view}/{label}: missing resized files: {list(sorted(missing_after_resize))[:10]}")

        if extra_after_resize:
            errors.append(f"{view}/{label}: extra resized files: {list(sorted(extra_after_resize))[:10]}")

        for fname in resized_files:
            path = os.path.join(resized_folder, fname)
            try:
                with Image.open(path) as img:
                    sizes_found.add(img.size)
                    if img.size != TARGET_SIZE:
                        errors.append(f"Wrong size: {path} -> {img.size}")
            except Exception as e:
                errors.append(f"Unreadable image: {path} | {e}")

print("POST-RESIZE VALIDATION\n")

print("Counts comparison:")
for key in sorted(counts_orig.keys()):
    print(f"{key}: original={counts_orig[key]}, resized={counts_resized[key]}")

print("\nSizes found in resized dataset:")
print(sizes_found)

print("\nErrors:")
if not errors:
    print("No errors found. Resize step is correct.")
else:
    for e in errors:
        print(e)