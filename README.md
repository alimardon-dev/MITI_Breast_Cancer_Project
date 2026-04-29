# 🧠 Breast Cancer Classification using CNN & Radiomics Fusion

## 📌 Overview

This project focuses on **breast cancer classification** using mammogram images from the **CBIS-DDSM dataset**.

We developed a **hybrid deep learning model** that combines:

* 🖼️ **CNN-based image features (ResNet18)**
* 📊 **PyRadiomics numerical features**

to improve classification performance between:

```text
Benign vs Malignant
```

---

## 🎯 Objective

> Build a robust and generalizable model by combining **multi-view mammograms (CC + MLO)** with **radiomics features**, while addressing:

* Data leakage
* Class imbalance
* Overfitting

---

## 🗂️ Dataset

* **Source:** CBIS-DDSM
* **Format:** DICOM → converted to PNG
* **Views:**

  * CC (Cranio-Caudal)
  * MLO (Medio-Lateral Oblique)

⚠️ Dataset is **not included** due to size limitations.

---

## ⚙️ Project Pipeline

```text
DICOM Images
     ↓
PNG Conversion
     ↓
Cropping (ROI)
     ↓
Patient-wise Splitting
     ↓
Radiomics Feature Extraction
     ↓
CNN Training (CC / MLO / Dual)
     ↓
Fusion Model (CNN + Radiomics)
     ↓
Evaluation & Comparison
```

---

## 🧠 Models Implemented

### 🔹 Single-view CNN

* CC-only model
* MLO-only model

### 🔹 Dual-input CNN

* Combines CC + MLO images

### 🔥 Fusion Model (Main Contribution)

* Combines:

  * CNN features (images)
  * Radiomics features (CSV)

---

## 🧪 Training Strategy

* Transfer Learning using **ResNet18**
* Partial layer freezing
* Data augmentation:

  * Horizontal flip
  * Rotation
* Class imbalance handling:

  * Weighted loss function
* Validation-based tuning

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall (important for cancer detection)
* F1-score
* Confusion Matrix
* ROC Curve

---

## 📈 Key Results

| Model                       | Performance      |
| --------------------------- | ---------------- |
| CC only                     | Moderate         |
| MLO only                    | Lower            |
| Dual CNN                    | Improved         |
| Dual + Augmentation         | Strong           |
| 🔥 Fusion (CNN + Radiomics) | **Best overall** |

---

## 🔍 Key Insights

* ✅ Dual-view improves classification performance
* ✅ Data augmentation reduces overfitting
* ✅ Class weights improve malignant detection
* ✅ Radiomics features enhance model understanding

---

## 📁 Project Structure

```text
MITI_BREAST_CANCER_PROJECT/
│
├── CNN_Models/
├── Fusion_Model/
├── Preprocessing/
├── Radiomics/
├── Evaluation/
├── RESULTS/
├── Scripts/
├── Presentation/
└── README.md
```

---

## 🚀 How to Run

### 1. Prepare Dataset

Download CBIS-DDSM dataset and place in:

```text
Data/Raw/
```

---

### 2. Preprocessing

```bash
python Preprocessing/convert_to_png.py
python Preprocessing/split_cc_patientwise.py
```

---

### 3. Radiomics Extraction

```bash
python Radiomics/radiomics_extract.py
```

---

### 4. Train CNN Models

```bash
python CNN_Models/train_dual.py
```

---

### 5. Train Fusion Model

```bash
python Fusion_Model/train_fusion.py
```

---

## 📌 Notes

* Large files (DICOM, PNG, models) are excluded via `.gitignore`
* Only essential code and results visualizations are included

---

## 🎓 Conclusion

This project demonstrates that combining:

```text
Deep Learning (CNN) + Traditional Features (Radiomics)
```

leads to a **more accurate and robust breast cancer classification system**.

---

## 💡 Future Work

* Use deeper models (ResNet50, EfficientNet)
* Apply attention mechanisms
* Improve fusion strategies
* Clinical validation

---

## 👨‍💻 Author

Alimardon 