import os

ROOT = "Data_Mass_Train_Cropped"

count = 0

for root, _, files in os.walk(ROOT):
    for f in files:
        print("FOUND:", f)
        count += 1
        if count > 20:
            break
    if count > 20:
        break

print("TOTAL FOUND:", count)