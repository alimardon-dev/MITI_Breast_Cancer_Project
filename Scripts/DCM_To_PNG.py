import pydicom
import numpy as np
from PIL import Image

# your DICOM file path
dcm_path = r"MalignantSamplePNG/Mass-Training_P_00001_LEFT_CC_1_1-1.dcm"

# output PNG path
png_path = r"MalignantSamplePNG/Mass-Training_P_00001_LEFT_CC_1_1-1.png"

# read DICOM
ds = pydicom.dcmread(dcm_path)
image = ds.pixel_array.astype(np.float32)

# normalize (VERY IMPORTANT)
image -= image.min()
if image.max() != 0:
    image /= image.max()
image *= 255.0

image = image.astype(np.uint8)

# convert to image and save
img = Image.fromarray(image)
img.save(png_path)

print("Converted successfully!")