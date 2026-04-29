import os
from collections import defaultdict

ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\Data_Mass_Train_Cropped_PNG_RESIZED"

VIEWS = ["CC", "MLO"]
LABELS = ["Benign", "Malignant"]


def extract_patient_id(filename):
    parts = filename.split("_")
    for i in range(len(parts) - 1):
        if parts[i] == "P":
            return f"P_{parts[i+1]}"
    return None


# store which views each patient has
patient_views = defaultdict(set)

for view in VIEWS:
    for label in LABELS:
        folder = os.path.join(ROOT, view, label)

        if not os.path.exists(folder):
            print(f"Missing folder: {folder}")
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".png"):
                continue

            pid = extract_patient_id(fname)
            if pid is None:
                print(f"Could not extract patient ID from: {fname}")
                continue

            patient_views[pid].add(view)


# classify patients
A = []  # CC + MLO
B = []  # CC only
C = []  # MLO only

for pid, views in patient_views.items():
    if "CC" in views and "MLO" in views:
        A.append(pid)
    elif "CC" in views:
        B.append(pid)
    elif "MLO" in views:
        C.append(pid)


# print results
print("=== CASE DISTRIBUTION ===\n")

print(f"A (CC + MLO): {len(A)}")
print(f"B (CC only):  {len(B)}")
print(f"C (MLO only): {len(C)}")

print(f"\nTotal patients: {len(patient_views)}")

# sanity check
print("\nCheck:")
print(f"A + B + C = {len(A) + len(B) + len(C)}")

# optional: show examples
print("\nExample A patients:", A[:5])
print("Example B patients:", B[:5])
print("Example C patients:", C[:5])