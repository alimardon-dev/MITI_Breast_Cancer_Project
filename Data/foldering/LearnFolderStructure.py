import os
from collections import defaultdict

CBIS_DDSM_PATH = r"C:\Users\USER\Desktop\MITI_Breast_Cancer_Project\Data\Raw\CBIS-DDSM"
OUTPUT_TXT = r"C:\Users\USER\Desktop\MITI_Breast_Cancer_Project\Data\foldering\folder_structure_analysis.txt"

# Only focus on Mass-Test and Mass-Training folders
ALLOWED_PREFIXES = ("Mass-Test_", "Mass-Training_")


def get_folder_structure(path, depth=0, max_depth=10):
    structure = {"_files": [], "_subdirs": {}}
    if depth > max_depth:
        return structure
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        return structure
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            structure["_subdirs"][item] = get_folder_structure(item_path, depth + 1, max_depth)
        else:
            structure["_files"].append(item)
    return structure


def generalize_name(name):
    import re
    name = re.sub(r'\d{2}-\d{2}-\d{4}-DDSM-NA-\d+', 'DD-MM-YYYY-DDSM-NA-NNNNN', name)
    name = re.sub(r'1\.000000-([\w\s]+)-\d+', r'1.000000-\1-NNNNN', name)
    name = re.sub(r'(Mass-Test|Mass-Training)_P_\d+_(LEFT|RIGHT)_(CC|MLO)(_\d+)?',
                  r'\1_P_NNNNN_\2_\3\4', name)
    return name


def get_pattern_lines(structure, depth=0):
    indent = "  " * depth
    lines = []
    for fname in structure["_files"]:
        lines.append(f"{indent}[FILE] {generalize_name(fname)}")
    for dname, sub in structure["_subdirs"].items():
        lines.append(f"{indent}[DIR]  {generalize_name(dname)}/")
        lines.extend(get_pattern_lines(sub, depth + 1))
    return lines


def main():
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log("CBIS-DDSM FOLDER STRUCTURE ANALYSIS REPORT")
    log("=" * 80)
    log(f"Source: {CBIS_DDSM_PATH}")
    log(f"Focused on: Mass-Test_ and Mass-Training_ folders only")
    log("=" * 80)

    if not os.path.exists(CBIS_DDSM_PATH):
        log(f"ERROR: Path not found: {CBIS_DDSM_PATH}")
        save(lines, OUTPUT_TXT)
        return

    all_folders = sorted(os.listdir(CBIS_DDSM_PATH))
    total_all = sum(1 for f in all_folders if os.path.isdir(os.path.join(CBIS_DDSM_PATH, f)))

    mass_folders = [
        f for f in all_folders
        if os.path.isdir(os.path.join(CBIS_DDSM_PATH, f))
        and f.startswith(ALLOWED_PREFIXES)
    ]
    ignored_folders = [
        f for f in all_folders
        if os.path.isdir(os.path.join(CBIS_DDSM_PATH, f))
        and not f.startswith(ALLOWED_PREFIXES)
    ]

    log(f"Total folders in CBIS-DDSM : {total_all}")
    log(f"  Mass-Test_ + Mass-Training_: {len(mass_folders)}")
    log(f"  Ignored (Calc-*, other)    : {len(ignored_folders)}")
    log("")

    # Group ignored by prefix for info
    from collections import Counter
    prefix_counts = Counter()
    for f in ignored_folders:
        prefix = f.split("_P_")[0] if "_P_" in f else f.split("_")[0]
        prefix_counts[prefix] += 1
    log("Ignored folder breakdown:")
    for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
        log(f"  {prefix}: {count}")
    log("")
    log("-" * 80)
    log(f"Scanning {len(mass_folders)} Mass folders...")
    log("-" * 80)

    # Learn patterns
    pattern_groups = defaultdict(list)

    for i, folder in enumerate(mass_folders):
        folder_path = os.path.join(CBIS_DDSM_PATH, folder)
        structure = get_folder_structure(folder_path)
        pattern_lines = get_pattern_lines(structure)
        pattern_key = "\n".join([generalize_name(l) for l in pattern_lines])
        pattern_groups[pattern_key].append(folder)

        if (i + 1) % 500 == 0:
            print(f"  Scanned {i+1}/{len(mass_folders)} folders...")

    log(f"Unique folder structures found: {len(pattern_groups)}")
    log("=" * 80)

    for rank, (pattern, folders) in enumerate(
        sorted(pattern_groups.items(), key=lambda x: -len(x[1])), start=1
    ):
        log("")
        log(f"PATTERN #{rank}  —  {len(folders)} folder(s)")
        log("-" * 60)
        log("Structure:")
        for pline in pattern.split("\n"):
            log(f"  {pline}")
        log("")
        log(f"Example folders ({min(3, len(folders))} shown):")
        for ex in folders[:3]:
            log(f"  - {ex}")
        if len(folders) > 3:
            log(f"  ... and {len(folders) - 3} more")
        log("")

    log("")
    log("=" * 80)
    log("SUMMARY TABLE")
    log("-" * 60)
    log(f"{'Pattern #':<12} {'Count':>8}    Structure (first line)")
    log("-" * 60)
    for rank, (pattern, folders) in enumerate(
        sorted(pattern_groups.items(), key=lambda x: -len(x[1])), start=1
    ):
        first_line = pattern.split("\n")[0] if pattern else "(empty)"
        log(f"  #{rank:<9} {len(folders):>8}    {first_line}")
    log("-" * 60)
    log(f"  {'TOTAL':<10} {len(mass_folders):>8}")
    log("=" * 80)

    save(lines, OUTPUT_TXT)


def save(lines, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()