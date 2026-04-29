import os
import pydicom
from PIL import Image

INPUT_TRAIN = "Data_Mass_Train_Cropped"
INPUT_TEST = "Data_Mass_Test_Cropped"

OUTPUT_TRAIN = "Data_Mass_Train_Cropped_PNG"
OUTPUT_TEST = "Data_Mass_Test_Cropped_PNG"


def convert(input_root, output_root):
    print(f"Converting {input_root} -> {output_root}")

    for root, _, files in os.walk(input_root):
        for f in files:
            if f.endswith(".dcm"):
                dcm_path = os.path.join(root, f)

                # keep same structure
                rel = os.path.relpath(root, input_root)
                out_dir = os.path.join(output_root, rel)
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(out_dir, f.replace(".dcm", ".png"))

                try:
                    dcm = pydicom.dcmread(dcm_path)
                    img = dcm.pixel_array

                    # normalize
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img = (img * 255).astype("uint8")

                    img = Image.fromarray(img)
                    img.save(out_path)

                except Exception as e:
                    print(f"ERROR: {dcm_path}")


if __name__ == "__main__":
    convert(INPUT_TRAIN, OUTPUT_TRAIN)
    convert(INPUT_TEST, OUTPUT_TEST)

    print("\nDONE converting to PNG")