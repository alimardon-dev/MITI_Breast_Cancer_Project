import os
from pathlib import Path

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "cc_distribution_report.txt"

CLASSES = ["Benign", "Malignant"]
SPLITS = ["train", "val", "test"]


def count_train_val(folder_path):
    return sum(
        1 for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )


def count_test(folder_path):
    count = 0

    for patient_path in folder_path.iterdir():
        if not patient_path.is_dir():
            continue

        cc_path = patient_path / "CC.png"

        if cc_path.exists():
            count += 1

    return count


def process_split(split):
    results = {}
    total = 0

    for cls in CLASSES:
        cls_path = BASE_DIR / split / cls

        if not cls_path.exists():
            print(f"Missing folder: {cls_path}")
            results[cls] = 0
            continue

        if split == "test":
            count = count_test(cls_path)
        else:
            count = count_train_val(cls_path)

        results[cls] = count
        total += count

    return results, total


def main():
    report_lines = []
    report_lines.append("CNN_CC_View Distribution Report")
    report_lines.append("=" * 40)
    report_lines.append("")

    overall_total = 0
    overall_counts = {"Benign": 0, "Malignant": 0}

    for split in SPLITS:
        results, total = process_split(split)

        benign = results["Benign"]
        malignant = results["Malignant"]

        overall_counts["Benign"] += benign
        overall_counts["Malignant"] += malignant
        overall_total += total

        benign_ratio = benign / total if total > 0 else 0
        malignant_ratio = malignant / total if total > 0 else 0

        report_lines.append(f"{split.upper()}")
        report_lines.append("-" * 20)
        report_lines.append(f"Benign: {benign}")
        report_lines.append(f"Malignant: {malignant}")
        report_lines.append(f"Total: {total}")
        report_lines.append(f"Benign Ratio: {benign_ratio:.3f}")
        report_lines.append(f"Malignant Ratio: {malignant_ratio:.3f}")
        report_lines.append("")

    report_lines.append("=" * 40)
    report_lines.append("OVERALL")
    report_lines.append("-" * 20)

    if overall_total > 0:
        overall_benign_ratio = overall_counts["Benign"] / overall_total
        overall_malignant_ratio = overall_counts["Malignant"] / overall_total
    else:
        overall_benign_ratio = 0
        overall_malignant_ratio = 0

    report_lines.append(f"Total Benign: {overall_counts['Benign']}")
    report_lines.append(f"Total Malignant: {overall_counts['Malignant']}")
    report_lines.append(f"Total Files: {overall_total}")
    report_lines.append(f"Benign Ratio: {overall_benign_ratio:.3f}")
    report_lines.append(f"Malignant Ratio: {overall_malignant_ratio:.3f}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("✅ Report saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    main()