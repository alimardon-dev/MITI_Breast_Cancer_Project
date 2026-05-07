import os

CBIS_DDSM_PATH = r"C:\Users\USER\Desktop\MITI_Breast_Cancer_Project\Data\Raw\CBIS-DDSM"
OUTPUT_TXT = r"C:\Users\USER\Desktop\MITI_Breast_Cancer_Project\Data\foldering\duplicate_folders_structure.txt"

# The 12 folders with 2+ subfolders
DUPLICATE_FOLDERS = [
    "Mass-Test_P_00145_LEFT_CC_1",
    "Mass-Test_P_00145_LEFT_MLO_1",
    "Mass-Test_P_00192_RIGHT_CC_1",
    "Mass-Test_P_00381_LEFT_CC_1",
    "Mass-Test_P_00381_LEFT_MLO_1",
    "Mass-Test_P_00699_RIGHT_CC_1",
    "Mass-Test_P_00699_RIGHT_MLO_1",
    "Mass-Test_P_00922_RIGHT_CC_1",
    "Mass-Test_P_00922_RIGHT_MLO_1",
    "Mass-Test_P_01183_LEFT_CC",
    "Mass-Test_P_01595_LEFT_CC_1",
    "Mass-Test_P_01595_LEFT_MLO_1",
]


def print_tree(path, lines, prefix=""):
    """Recursively print folder/file tree."""
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        lines.append(f"{prefix}  [Permission Denied]")
        return

    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        connector = "└── " if i == len(items) - 1 else "├── "
        extension = "    " if i == len(items) - 1 else "│   "

        if os.path.isdir(item_path):
            log(f"{prefix}{connector}[FOLDER] {item}/", lines)
            print_tree(item_path, lines, prefix + extension)
        else:
            size = os.path.getsize(item_path)
            log(f"{prefix}{connector}[FILE]   {item}  ({size:,} bytes)", lines)


def log(text, lines):
    print(text)
    lines.append(text)


def main():
    lines = []

    log("DUPLICATE SUBFOLDER STRUCTURE REPORT", lines)
    log("=" * 80, lines)
    log(f"Source: {CBIS_DDSM_PATH}", lines)
    log(f"Showing {len(DUPLICATE_FOLDERS)} folders with 2+ subfolders", lines)
    log("=" * 80, lines)

    for folder_name in DUPLICATE_FOLDERS:
        folder_path = os.path.join(CBIS_DDSM_PATH, folder_name)
        log("", lines)
        log("=" * 80, lines)
        log(f"FOLDER: {folder_name}", lines)
        log("=" * 80, lines)

        if not os.path.exists(folder_path):
            log(f"  WARNING: FOLDER NOT FOUND: {folder_path}", lines)
            continue

        subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        log(f"  Subfolders inside: {len(subfolders)}", lines)
        log("", lines)

        print_tree(folder_path, lines, prefix="  ")

    log("", lines)
    log("=" * 80, lines)
    log(f"Report complete. {len(DUPLICATE_FOLDERS)} folders shown.", lines)
    log("=" * 80, lines)

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
