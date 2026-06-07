# PPE Compliance Detection using CNN

A deep learning project that detects whether workers on a construction site are wearing proper Personal Protective Equipment (hard hat and safety vest). Built as part of the Machine Learning course at the University of Europe for Applied Sciences.

---

## What this does

Takes an image of a worker and classifies it into one of four categories:

- **PPE_Compliant** — wearing both hard hat and safety vest
- **HardHat_Only** — only wearing a hard hat
- **Vest_Only** — only wearing a safety vest
- **No_PPE** — wearing neither

Three models are trained and compared: a custom CNN built from scratch, MobileNetV2, and ResNet50.

---

## Dataset

Using the [Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) from Kaggle (~2000 images across the 4 classes).

---

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/pavanpitta/PPE-Compliance-Detection-System.git
cd PPE-Compliance-Detection-System
pip install -r requirements.txt
```

Download the dataset via Kaggle CLI:

```bash
kaggle datasets download -d snehilsanyal/construction-site-safety-image-dataset-roboflow
unzip construction-site-safety-image-dataset-roboflow.zip -d data/
```

Then organize images into this structure:

```
data/ppe_dataset/
    PPE_Compliant/
    HardHat_Only/
    Vest_Only/
    No_PPE/
```

Update `DATASET_ROOT` in the script to point to this folder, then run:

```bash
python ppe_cnn_implementation.py
```

Everything gets saved to the `outputs/` folder automatically.

---

## Project Structure

```
ppe-compliance-detection/
├── ppe_cnn_implementation.py
├── requirements.txt
├── README.md
├── data/
│   └── ppe_dataset/
└── outputs/
```

---

## Models

| Model | Details |
|---|---|
| Custom CNN | 4 conv blocks, BatchNorm, Dropout, trained from scratch |
| MobileNetV2 | ImageNet weights, fine-tuned on upper blocks |
| ResNet50 | ImageNet weights, fine-tuned on last 20 layers |

---

## Results (expected)

| Model | Test Accuracy | Macro AUC |
|---|---|---|
| Custom CNN | ~78% | ~0.90 |
| MobileNetV2 | ~91% | ~0.96 |
| ResNet50 | ~92% | ~0.97 |

---

## Output files

After running the script you'll find:

- Class distribution chart
- Sample image grid per class
- Training/validation accuracy and loss curves for all 3 models
- Confusion matrices (absolute + normalized)
- ROC curves per class
- Grad-CAM heatmaps for correct and wrong predictions
- Top misclassification examples
- Classification reports as `.txt` files
- Saved model files (`.keras`)

---

## Dependencies

```
tensorflow>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
Pillow>=9.0.0
```

---

## Notes

- If running on Kaggle, change `DATASET_ROOT` to the `/kaggle/input/` path and set `OUTPUT_DIR` to `/kaggle/working/`
- GPU recommended — training all 3 models takes around 30–60 minutes depending on hardware
- For Grad-CAM on the custom CNN, check the last activation layer name from `model.summary()` if you get a layer not found error

---

**Author:** Pavan Pitta — University of Europe for Applied Sciences  
**Supervisor:** Raja Hasim Ali
