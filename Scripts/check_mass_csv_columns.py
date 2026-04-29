import pandas as pd

train_csv = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\csv files\mass_case_description_train_set.csv"
test_csv = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\csv files\mass_case_description_test_set.csv"

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

print("TRAIN COLUMNS:")
for col in df_train.columns:
    print(repr(col))

print("\n" + "="*60 + "\n")

print("TEST COLUMNS:")
for col in df_test.columns:
    print(repr(col))