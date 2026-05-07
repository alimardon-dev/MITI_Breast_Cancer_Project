import os

CBIS_DDSM_PATH = r"C:\Users\USER\Desktop\MITI_Breast_Cancer_Project\Data\Raw\CBIS-DDSM"
OUTPUT_TXT = r"C:\Users\USER\Desktop\MITI_Breast_Cancer_Project\Data\foldering\CBIS_DDSM_folder_report.txt"


def count_folders_and_files(root_path, output_path):
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log(f"Scanning: {root_path}")
    log("=" * 80)

    if not os.path.exists(root_path):
        log(f"ERROR: Path does not exist: {root_path}")
        save(lines, output_path)
        return

    # Get all top-level folders that start with "Mass-Test_P_"
    all_items = sorted(os.listdir(root_path))
    mass_test_folders = [
        f for f in all_items
        if os.path.isdir(os.path.join(root_path, f)) and f.startswith("Mass-Test_P_")
    ]
    other_folders = [
        f for f in all_items
        if os.path.isdir(os.path.join(root_path, f)) and not f.startswith("Mass-Test_P_")
    ]

    log(f"Total folders in CBIS-DDSM        : {len(all_items)}")
    log(f"  Mass-Test_P_ folders             : {len(mass_test_folders)}")
    log(f"  Other folders (ignored)          : {len(other_folders)}")
    log("-" * 80)

    total_date_subfolders = 0
    total_files = 0
    folders_with_1_sub = 0
    folders_with_2_sub = 0
    folders_with_0_sub = 0

    for folder in mass_test_folders:
        folder_path = os.path.join(root_path, folder)

        # Level 2: date-stamped subfolders (e.g. 10-04-2016-DDSM-NA-25244)
        date_subfolders = sorted([
            d for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        ])

        n_subs = len(date_subfolders)
        total_date_subfolders += n_subs

        if n_subs == 0:
            folders_with_0_sub += 1
        elif n_subs == 1:
            folders_with_1_sub += 1
        else:
            folders_with_2_sub += 1

        log(f"\nFolder: {folder}")
        log(f"  Subfolders ({n_subs}):")

        for sub in date_subfolders:
            sub_path = os.path.join(folder_path, sub)

            # Count files inside this date subfolder
            files = sorted([
                f for f in os.listdir(sub_path)
                if os.path.isfile(os.path.join(sub_path, f))
            ])
            total_files += len(files)

            log(f"    [{len(files)} file(s)] {sub}")
            for fname in files:
                log(f"      - {fname}")

    log("")
    log("=" * 80)
    log("SUMMARY")
    log(f"  Mass-Test_P_ top-level folders   : {len(mass_test_folders)}")
    log(f"  Other folders (ignored)          : {len(other_folders)}")
    log(f"  Total date subfolders            : {total_date_subfolders}")
    log(f"    Folders with 0 subfolders      : {folders_with_0_sub}")
    log(f"    Folders with 1 subfolder       : {folders_with_1_sub}")
    log(f"    Folders with 2+ subfolders     : {folders_with_2_sub}")
    log(f"  Total files (all .dcm etc.)      : {total_files}")
    log("=" * 80)

    save(lines, output_path)


def save(lines, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    count_folders_and_files(CBIS_DDSM_PATH, OUTPUT_TXT)