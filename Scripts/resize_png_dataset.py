import os
from PIL import Image

# ===== PATHS =====
SRC_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\Data_Mass_Train_Cropped_PNG"
DST_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\Data_Mass_Train_Cropped_PNG_RESIZED"

TARGET_SIZE = (224, 224)

def process():
    print("Starting resizing...")
    print(f"Source: {SRC_ROOT}")
    print(f"Destination: {DST_ROOT}")
    print(f"Target size: {TARGET_SIZE}")

    for view in ["CC", "MLO"]:
        for label in ["Benign", "Malignant"]:
            src_folder = os.path.join(SRC_ROOT, view, label)
            dst_folder = os.path.join(DST_ROOT, view, label)

            if not os.path.exists(src_folder):
                print(f"Skipping missing: {src_folder}")
                continue

            os.makedirs(dst_folder, exist_ok=True)

            for fname in os.listdir(src_folder):
                if not fname.lower().endswith(".png"):
                    continue

                src_path = os.path.join(src_folder, fname)
                dst_path = os.path.join(dst_folder, fname)

                try:
                    with Image.open(src_path) as img:
                        img = img.convert("L")  # keep grayscale
                        img = img.resize(TARGET_SIZE)
                        img.save(dst_path)

                except Exception as e:
                    print(f"ERROR: {src_path} -> {e}")

    print("\nDone resizing.")


if __name__ == "__main__":
    process()