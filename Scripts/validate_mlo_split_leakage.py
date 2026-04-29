import os
from collections import defaultdict

ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\cnn_data_mlo"
VIEW = "MLO"
LABELS = ["Benign", "Malignant"]

def extract_patient_id(filename):
    parts = filename.split("_")
    for i in range(len(parts) - 1):
        if parts[i] == "P":
            return f"P_{parts[i+1]}"
    return None

patient_to_splits = defaultdict(set)
counts = defaultdict(int)
errors = []

for split in ["train", "val", "test"]:
    for label in LABELS:
        folder = os.path.join(ROOT, split, VIEW, label)

        if not os.path.exists(folder):
            errors.append(f"Missing folder: {folder}")
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".png"):
                continue

            pid = extract_patient_id(fname)
            if pid is None:
                errors.append(f"Could not extract patient ID from: {fname}")
                continue

            patient_to_splits[pid].add(split)
            counts[(split, label)] += 1

leakage = []
for pid, splits in patient_to_splits.items():
    if len(splits) > 1:
        leakage.append((pid, sorted(list(splits))))

print("MLO SPLIT VALIDATION\n")

print("Image counts:")
for key in sorted(counts.keys()):
    print(f"{key}: {counts[key]}")

print(f"\nTotal unique patients: {len(patient_to_splits)}")

print("\nLeakage check:")
if not leakage:
    print("No patient leakage found.")
else:
    print("Patient leakage detected:")
    for pid, splits in leakage[:20]:
        print(f"{pid}: {splits}")

print("\nOther errors:")
if not errors:
    print("No other errors found.")
else:
    for e in errors:
        print(e)