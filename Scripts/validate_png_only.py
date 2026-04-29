import os
from PIL import Image
from collections import defaultdict

# 🔥 CHANGE THIS TO YOUR REAL PATH
ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\Data_Mass_Train_Cropped_PNG"

sizes = set()
counts = defaultdict(int)
errors = []

# possible variations (handles CC/cc, Benign/benign etc.)
VIEWS = ["CC", "cc", "MLO", "mlo"]
LABELS = ["Benign", "benign", "Malignant", "malignant"]

print("Starting validation...\n")

found_any = False

for view in VIEWS:
    view_path = os.path.join(ROOT, view)
    if not os.path.exists(view_path):
        continue

    for label in LABELS:
        label_path = os.path.join(view_path, label)

        if not os.path.exists(label_path):
            continue

        print(f"Checking: {label_path}")
        found_any = True

        for f in os.listdir(label_path):
            path = os.path.join(label_path, f)

            # ✅ PNG check
            if not f.lower().endswith(".png"):
                errors.append(f"Not PNG: {path}")
                continue

            # ✅ Try opening image
            try:
                img = Image.open(path)
                sizes.add(img.size)
            except Exception as e:
                errors.append(f"Corrupted image: {path} | {e}")

            # ✅ Count
            counts[(view.upper(), label.capitalize())] += 1

if not found_any:
    print("❌ ERROR: No valid folders found. Check ROOT path.")
    exit()

# 📏 Image sizes
print("\n📏 Image sizes found:")
print(sizes)

# 📊 Counts
print("\n📊 Counts:")
for k, v in sorted(counts.items()):
    print(f"{k}: {v}")

# 🚨 Errors
print("\n🚨 Errors:")
if not errors:
    print("✅ No errors found. Dataset looks clean.")
else:
    for e in errors:
        print(e)