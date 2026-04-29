import os
import pandas as pd

base_folder = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\manifest-ZkhPvrLo5216730872708713142\project\extracted_features_cropped\Mass_train"

cc_input = os.path.join(base_folder, "cc_extracted_features.csv")
mlo_input = os.path.join(base_folder, "mlo_extracted_features.csv")

cc_output = os.path.join(base_folder, "cc_extracted_features_without_shape.csv")
mlo_output = os.path.join(base_folder, "mlo_extracted_features_without_shape.csv")


def remove_shape_features(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    shape_cols = [col for col in df.columns if col.startswith("original_shape2D_")]

    df_no_shape = df.drop(columns=shape_cols)

    df_no_shape.to_csv(output_csv, index=False)

    print("=" * 50)
    print(f"Processed: {os.path.basename(input_csv)}")
    print(f"Original columns : {len(df.columns)}")
    print(f"Removed shape cols: {len(shape_cols)}")
    print(f"New columns      : {len(df_no_shape.columns)}")


remove_shape_features(cc_input, cc_output)
remove_shape_features(mlo_input, mlo_output)

print("\nDONE")