import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATH
# =========================================================

base_folder = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\manifest-ZkhPvrLo5216730872708713142\project\extracted_features_cropped\Mass_train"

files = [
    "cc_extracted_features.csv",
    "cc_extracted_features_without_shape.csv",
    "mlo_extracted_features.csv",
    "mlo_extracted_features_without_shape.csv"
]

# =========================================================
# NORMALIZATION
# =========================================================

for file in files:
    path = os.path.join(base_folder, file)
    df = pd.read_csv(path)

    print(f"\nProcessing: {file}")

    # Metadata columns (DO NOT TOUCH)
    meta_cols = [
        "case_id",
        "file_name",
        "label",
        "class_name",
        "view",
        "width",
        "height",
        "lesion_pixels",
        "image_path",
        "mask_path"
    ]

    # Feature columns
    feature_cols = [c for c in df.columns if c not in meta_cols]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save
    output_path = path.replace(".csv", "_normalized.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Features normalized: {len(feature_cols)}")

print("\nDONE 🚀")