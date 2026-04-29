import os
from collections import defaultdict
from PIL import Image

# ====== CHANGE THIS IF NEEDED ======
BASE_DIR = r"C:\Users\USER\Desktop\CNN_training_breast_cancer"
DICOM_ROOT = os.path.join(BASE_DIR, "Data_Mass_Train_Cropped")
PNG_ROOT   = os.path.join(BASE_DIR, "Data_Mass_Train_Cropped_PNG")

# If your folder names are lowercase, keep these as shown.
VIEWS = ["CC", "MLO", "cc", "mlo"]
LABELS = ["Benign", "Malignant", "benign", "malignant"]


def find_existing_subfolders(root):
    found = []
    for view in VIEWS:
        for label in LABELS:
            path = os.path.join(root, view, label)
            if os.path.exists(path):
                found.append((view, label, path))
    return found


def normalize_base_name(filename):
    """
    Convert:
      something.dcm -> something
      something.png -> something
    """
    lower = filename.lower()
    if lower.endswith(".dcm"):
        return filename[:-4]
    if lower.endswith(".png"):
        return filename[:-4]
    return filename


def expected_view_in_name(filename):
    name = filename.upper()
    if "_CC_" in name:
        return "CC"
    if "_MLO_" in name:
        return "MLO"
    return None


def expected_label_in_name(filename):
    name = filename.lower()
    if "malig" in name:
        return "malignant"
    if "benign" in name:
        return "benign"
    return None


def get_files(folder):
    try:
        return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    except Exception:
        return []


def validate_pair_folder(dic_folder, png_folder, view, label, report):
    dic_files = get_files(dic_folder)
    png_files = get_files(png_folder)

    dic_bases = set()
    png_bases = set()

    dic_non_dcm = []
    png_non_png = []
    png_corrupt = []
    filename_view_mismatch = []
    filename_label_mismatch = []

    for f in dic_files:
        if not f.lower().endswith(".dcm"):
            dic_non_dcm.append(f)
        else:
            dic_bases.add(normalize_base_name(f))

        # filename sanity check
        inferred_view = expected_view_in_name(f)
        if inferred_view and inferred_view.upper() != view.upper():
            filename_view_mismatch.append(f"[DICOM] {f} says {inferred_view}, folder says {view}")

        inferred_label = expected_label_in_name(f)
        if inferred_label and inferred_label.lower() != label.lower():
            filename_label_mismatch.append(f"[DICOM] {f} says {inferred_label}, folder says {label}")

    png_sizes = set()

    for f in png_files:
        if not f.lower().endswith(".png"):
            png_non_png.append(f)
        else:
            png_bases.add(normalize_base_name(f))
            path = os.path.join(png_folder, f)
            try:
                with Image.open(path) as img:
                    png_sizes.add(img.size)
            except Exception as e:
                png_corrupt.append(f"{f} | {e}")

        # filename sanity check
        inferred_view = expected_view_in_name(f)
        if inferred_view and inferred_view.upper() != view.upper():
            filename_view_mismatch.append(f"[PNG] {f} says {inferred_view}, folder says {view}")

        inferred_label = expected_label_in_name(f)
        if inferred_label and inferred_label.lower() != label.lower():
            filename_label_mismatch.append(f"[PNG] {f} says {inferred_label}, folder says {label}")

    missing_png = sorted(dic_bases - png_bases)
    extra_png = sorted(png_bases - dic_bases)

    report["counts"][(view.upper(), label.capitalize(), "DICOM")] = len([f for f in dic_files if f.lower().endswith(".dcm")])
    report["counts"][(view.upper(), label.capitalize(), "PNG")] = len([f for f in png_files if f.lower().endswith(".png")])

    report["folder_reports"].append({
        "view": view,
        "label": label,
        "dic_folder": dic_folder,
        "png_folder": png_folder,
        "dic_non_dcm": dic_non_dcm,
        "png_non_png": png_non_png,
        "png_corrupt": png_corrupt,
        "missing_png": missing_png,
        "extra_png": extra_png,
        "png_sizes": png_sizes,
        "filename_view_mismatch": filename_view_mismatch,
        "filename_label_mismatch": filename_label_mismatch,
    })


def main():
    print("SOURCE-DATA VALIDATION START\n")

    print("DICOM root:", DICOM_ROOT)
    print("PNG root:  ", PNG_ROOT)
    print()

    if not os.path.exists(DICOM_ROOT):
        print("ERROR: DICOM root does not exist.")
        return

    if not os.path.exists(PNG_ROOT):
        print("ERROR: PNG root does not exist.")
        return

    dic_subfolders = find_existing_subfolders(DICOM_ROOT)
    png_subfolders = find_existing_subfolders(PNG_ROOT)

    dic_keys = {(v.lower(), l.lower()) for v, l, _ in dic_subfolders}
    png_keys = {(v.lower(), l.lower()) for v, l, _ in png_subfolders}

    print("Found DICOM subfolders:")
    for v, l, p in dic_subfolders:
        print(f"  {v}/{l} -> {p}")

    print("\nFound PNG subfolders:")
    for v, l, p in png_subfolders:
        print(f"  {v}/{l} -> {p}")

    print("\nFOLDER STRUCTURE CHECK")
    missing_png_folders = sorted(dic_keys - png_keys)
    extra_png_folders = sorted(png_keys - dic_keys)

    if not missing_png_folders and not extra_png_folders:
        print("Folder structure matches.")
    else:
        if missing_png_folders:
            print("Missing PNG-side folders:")
            for item in missing_png_folders:
                print(" ", item)
        if extra_png_folders:
            print("Extra PNG-side folders:")
            for item in extra_png_folders:
                print(" ", item)

    report = {
        "counts": defaultdict(int),
        "folder_reports": []
    }

    common_keys = sorted(dic_keys & png_keys)

    print("\nPAIR MATCH VALIDATION\n")
    for view_key, label_key in common_keys:
        # recover actual folder names as they exist
        dic_view, dic_label, dic_path = next(
            (v, l, p) for (v, l, p) in dic_subfolders if v.lower() == view_key and l.lower() == label_key
        )
        png_view, png_label, png_path = next(
            (v, l, p) for (v, l, p) in png_subfolders if v.lower() == view_key and l.lower() == label_key
        )

        print(f"Checking {dic_view}/{dic_label}")
        validate_pair_folder(dic_path, png_path, dic_view, dic_label, report)

    print("\nCOUNTS SUMMARY")
    for key in sorted(report["counts"].keys()):
        print(f"{key}: {report['counts'][key]}")

    print("\nDETAILED ERRORS / WARNINGS")
    any_issue = False

    all_png_sizes = set()

    for fr in report["folder_reports"]:
        folder_has_issue = any([
            fr["dic_non_dcm"],
            fr["png_non_png"],
            fr["png_corrupt"],
            fr["missing_png"],
            fr["extra_png"],
            fr["filename_view_mismatch"],
            fr["filename_label_mismatch"],
        ])

        for s in fr["png_sizes"]:
            all_png_sizes.add(s)

        print(f"\n--- {fr['view']}/{fr['label']} ---")
        print("DICOM folder:", fr["dic_folder"])
        print("PNG folder:  ", fr["png_folder"])
        print("PNG sizes:   ", fr["png_sizes"] if fr["png_sizes"] else "None found")

        if fr["dic_non_dcm"]:
            any_issue = True
            print("Non-DICOM files in DICOM folder:")
            for x in fr["dic_non_dcm"][:20]:
                print("  ", x)

        if fr["png_non_png"]:
            any_issue = True
            print("Non-PNG files in PNG folder:")
            for x in fr["png_non_png"][:20]:
                print("  ", x)

        if fr["png_corrupt"]:
            any_issue = True
            print("Corrupted/unreadable PNG files:")
            for x in fr["png_corrupt"][:20]:
                print("  ", x)

        if fr["missing_png"]:
            any_issue = True
            print("DICOM files with NO matching PNG:")
            for x in fr["missing_png"][:20]:
                print("  ", x)
            if len(fr["missing_png"]) > 20:
                print(f"  ... and {len(fr['missing_png']) - 20} more")

        if fr["extra_png"]:
            any_issue = True
            print("PNG files with NO matching DICOM:")
            for x in fr["extra_png"][:20]:
                print("  ", x)
            if len(fr["extra_png"]) > 20:
                print(f"  ... and {len(fr['extra_png']) - 20} more")

        if fr["filename_view_mismatch"]:
            any_issue = True
            print("Filename/folder view mismatches:")
            for x in fr["filename_view_mismatch"][:20]:
                print("  ", x)

        if fr["filename_label_mismatch"]:
            any_issue = True
            print("Filename/folder label mismatches:")
            for x in fr["filename_label_mismatch"][:20]:
                print("  ", x)

        if not folder_has_issue:
            print("No issues found in this folder.")

    print("\nGLOBAL PNG SIZE SUMMARY")
    print(all_png_sizes if all_png_sizes else "No PNG sizes found")

    print("\nFINAL SOURCE-DATA STATUS")
    if not any_issue and not missing_png_folders and not extra_png_folders:
        print("PASS: Source-data level looks consistent.")
        print("You are ready for the NEXT stage: build train/val/test CNN dataset.")
    else:
        print("CHECK REQUIRED: Source-data level has mismatches or inconsistencies.")
        print("Fix these before building the final CNN split.")


if __name__ == "__main__":
    main()