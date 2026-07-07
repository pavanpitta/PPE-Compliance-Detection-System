import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------- Workflow diagram ----------
fig, ax = plt.subplots(figsize=(15, 3.2))
ax.set_xlim(0, 16); ax.set_ylim(0, 3); ax.axis('off')
stages = ['Dataset\nAcquisition\n(Roboflow CSS)', 'Person-crop\nExtraction\n(YOLO boxes)',
          'Label\nDerivation\n(4 PPE classes)', 'Preprocess +\nStratified Split\n70/15/15',
          'Class\nWeighting', 'Train Compact\n& Deeper CNN', 'Evaluation\n(Acc/F1/AUC)',
          'Grad-CAM\nInterpretation']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#4C72B0', '#55A868']
n = len(stages); w = 1.7; gap = (16 - n*w) / (n+1)
for i, (s, c) in enumerate(zip(stages, colors)):
    x = gap + i*(w+gap)
    box = FancyBboxPatch((x, 1.05), w, .95, boxstyle='round,pad=0.03,rounding_size=0.1',
                         fc=c, ec='black', lw=1, alpha=.9)
    ax.add_patch(box)
    ax.text(x+w/2, 1.52, s, ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax.text(x+w/2, 2.15, str(i+1), ha='center', fontsize=10, fontweight='bold')
    if i < n-1:
        ax.add_patch(FancyArrowPatch((x+w, 1.52), (x+w+gap, 1.52),
                     arrowstyle='-|>', mutation_scale=15, lw=1.5, color='#333'))
plt.tight_layout(); plt.savefig('Figures/workflow.pdf'); plt.close()

# ---------- CNN architecture diagram ----------
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 18); ax.set_ylim(0, 4); ax.axis('off')
blocks = [('Input\n96x96x3', '#DDD', .8),
          ('Conv 32\n+BN+ReLU', '#4C72B0', .9),
          ('Pool\n48x48', '#B0C4DE', .5),
          ('Conv 64\n+BN+ReLU', '#4C72B0', .9),
          ('Pool\n24x24', '#B0C4DE', .5),
          ('Conv 128\n+BN+ReLU', '#4C72B0', .9),
          ('Pool\n12x12', '#B0C4DE', .5),
          ('Conv 256\n+BN+ReLU', '#4C72B0', .9),
          ('Pool\n6x6', '#B0C4DE', .5),
          ('Global\nAvgPool', '#55A868', .8),
          ('Dense 128\n+Drop 0.3', '#CCB974', .8),
          ('Softmax\n4 classes', '#C44E52', .8)]
x = .3
for name, c, w in blocks:
    h = 1.6
    box = FancyBboxPatch((x, 1.2), w, h, boxstyle='round,pad=0.02,rounding_size=0.05',
                         fc=c, ec='black', lw=.8)
    ax.add_patch(box)
    ax.text(x+w/2, 2.0, name, ha='center', va='center', fontsize=7,
            color='white' if c in ('#4C72B0', '#55A868', '#C44E52') else 'black', fontweight='bold')
    if x > .3:
        ax.add_patch(FancyArrowPatch((x-.15, 2.0), (x, 2.0), arrowstyle='-|>', mutation_scale=10, lw=1, color='#555'))
    x += w + .35
ax.text(9, 3.6, 'Compact Custom CNN Architecture (~1.6M parameters)', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.savefig('Figures/architecture.pdf'); plt.close()

# ---------- graphical abstract ----------
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis('off')
ax.text(6, 3.6, 'PPE Compliance Detection from Construction-Site Imagery',
        ha='center', fontsize=13, fontweight='bold')
steps = [('Construction\nCCTV image', '#DDD'), ('Detect &\ncrop person', '#4C72B0'),
         ('CNN\nclassifier', '#55A868'), ('4 PPE\ncompliance\nclasses', '#C44E52'),
         ('Grad-CAM\nexplanation', '#CCB974')]
x = 0.5
for name, c in steps:
    box = FancyBboxPatch((x, 1.3), 1.8, 1.4, boxstyle='round,pad=0.03,rounding_size=0.1',
                         fc=c, ec='black', lw=1, alpha=.9)
    ax.add_patch(box)
    ax.text(x+.9, 2.0, name, ha='center', va='center', fontsize=8.5,
            color='white' if c in ('#4C72B0', '#55A868', '#C44E52') else 'black', fontweight='bold')
    if x > .5:
        ax.add_patch(FancyArrowPatch((x-.35, 2.0), (x, 2.0), arrowstyle='-|>', mutation_scale=15, lw=1.5, color='#333'))
    x += 2.3
ax.text(6, .6, 'Best model: Compact CNN — 74.5% test accuracy, 0.90 macro-AUC',
        ha='center', fontsize=10, style='italic', color='#333')
plt.tight_layout(); plt.savefig('Figures/graphical_abstract.pdf'); plt.close()
print('diagrams done')
