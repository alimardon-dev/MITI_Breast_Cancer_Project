import os
import pydicom
import numpy as np
from PIL import Image

SRC_ROOT = r"C:\Users\USER\Desktop\CNN_Training_2.0\Data_Mass_Test_Cropped"
DST_ROOT = r"C:\Users\USER\Desktop\CNN_Training_2.0\Data_Mass_Test_Cropped_PNG"

def normalize_to_uint8(image):
    image = image.astype(np.float32)
    image -= image.min()
    if image.max() != 0:
        image /= image.max()
    image *= 255.0
    return image.astype(np.uint8)

def dicom_to_png(dcm_path, png_path):
    try:
        dcm = pydicom.dcmread(dcm_path)
        image = dcm.pixel_array

        image = normalize_to_uint8(image)

        img = Image.fromarray(image)
        img = img.convert("L")  # grayscale

        img.save(png_path)

    except Exception as e:
        print(f"Error: {dcm_path} -> {e}")

def process_folder():
    for root, dirs, files in os.walk(SRC_ROOT):
        for file in files:
            if not file.lower().endswith(".dcm"):
                continue

            dcm_path = os.path.join(root, file)

            # preserve folder structure
            relative_path = os.path.relpath(root, SRC_ROOT)
            save_folder = os.path.join(DST_ROOT, relative_path)
            os.makedirs(save_folder, exist_ok=True)

            png_name = file.replace(".dcm", ".png")
            png_path = os.path.join(save_folder, png_name)

            dicom_to_png(dcm_path, png_path)

    print("\n✅ All test DICOM images converted to PNG!")

if __name__ == "__main__":
    process_folder()