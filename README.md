# PPE Compliance Detection System — Code

Worker-level PPE compliance classification (4 classes: PPE_Compliant, HardHat_Only, Vest_Only, NO_PPE)
derived from the Roboflow Construction Site Safety image dataset (YOLO format).

## Pipeline
1. `build_crops.py`   — Extract person-centred crops from YOLO boxes and derive 4-class compliance labels.
2. `run_model.py`     — Train the compact custom CNN (~0.42M params).
3. `run_deep.py` / `resume_deep.py` — Train the deeper custom CNN (~2.5M params).
4. `eval_model.py`    — Evaluate a saved checkpoint on the held-out test set.
5. `make_figs.py`     — Generate class distribution, training curves, confusion matrices, ROC, comparison.
6. `make_gradcam.py`  — Generate Grad-CAM overlays.
7. `make_diagrams.py` — Generate workflow, architecture, graphical-abstract diagrams.

## Results (held-out test set, 1120 crops)
| Model       | Accuracy | Macro F1 | Macro AUC | Params |
|-------------|----------|----------|-----------|--------|
| Compact CNN | 74.5%    | 0.686    | 0.902     | ~0.42M |
| Deeper CNN  | 70.9%    | 0.661    | 0.888     | ~2.5M  |

## Environment
Python 3.10+, TensorFlow 2.x, scikit-learn, Pillow, Matplotlib.
Trained from scratch (no ImageNet pre-trained weights).
