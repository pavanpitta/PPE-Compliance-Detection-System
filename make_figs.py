import json, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import PIL.Image as Image

os.makedirs('Figures', exist_ok=True)
CL = ['PPE_Compliant', 'HardHat_Only', 'Vest_Only', 'NO_PPE']
SHORT = ['PPE_Compliant', 'HardHat_Only', 'Vest_Only', 'NO_PPE']
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans', 'axes.grid': True,
                     'grid.alpha': .3, 'axes.axisbelow': True})
BLUE, ORANGE, GREEN, RED = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'

crop = json.load(open('crop_labels.json'))
resC = json.load(open('res_CustomCNN.json'))
resD = json.load(open('res_DeepCNN.json'))
histC = json.load(open('hist_CustomCNN.json'))
histD = json.load(open('hist_DeepCNN.json'))
split = json.load(open('split.json')) if os.path.exists('split.json') else None

# ---- Fig 1: class distribution (full train split) ----
import collections
tr_counts = collections.Counter([r[1] for r in crop])
fig, ax = plt.subplots(figsize=(7, 4.2))
vals = [tr_counts[c] for c in CL]
bars = ax.bar(SHORT, vals, color=[GREEN, ORANGE, BLUE, RED], edgecolor='black', linewidth=.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 30, str(v), ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of person-instances')
ax.set_title('PPE Compliance Class Distribution (person crops)')
ax.set_ylim(0, max(vals)*1.15)
plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig('Figures/class_distribution.pdf'); plt.close()

# ---- Fig 2: sample image grid 4x4 ----
fig, axes = plt.subplots(4, 4, figsize=(9, 9))
byc = {c: [r[0] for r in crop if r[1] == c] for c in CL}
for i, c in enumerate(CL):
    imgs = byc[c][:4]
    for j in range(4):
        ax = axes[i, j]
        if j < len(imgs):
            ax.imshow(Image.open(imgs[j]).resize((96, 96)))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if j == 0:
            ax.set_ylabel(c, fontsize=10, fontweight='bold')
axes[0, 0].set_title('Example 1'); axes[0, 1].set_title('Example 2')
axes[0, 2].set_title('Example 3'); axes[0, 3].set_title('Example 4')
plt.suptitle('Representative person crops per PPE compliance class', fontweight='bold')
plt.tight_layout(); plt.savefig('Figures/sample_images.pdf'); plt.close()

# ---- training curves helper ----
def train_fig(hist, fname, title):
    acc, val = hist['accuracy'], hist['val_accuracy']
    ep = range(1, len(acc)+1)
    has_loss = 'loss' in hist and len(hist['loss']) == len(acc)
    if has_loss:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    else:
        fig, a1 = plt.subplots(1, 1, figsize=(6.5, 4.5))
    a1.plot(ep, acc, '-o', color=BLUE, ms=3, label='Training')
    a1.plot(ep, val, '-o', color=ORANGE, ms=3, label='Validation')
    a1.set_xlabel('Epoch'); a1.set_ylabel('Accuracy'); a1.legend()
    a1.set_title(f'{title} — Accuracy'); a1.set_ylim(0, 1)
    if has_loss:
        a2.plot(ep, hist['loss'], '-o', color=BLUE, ms=3, label='Training')
        a2.plot(ep, hist['val_loss'], '-o', color=ORANGE, ms=3, label='Validation')
        a2.set_ylabel('Loss'); a2.set_xlabel('Epoch'); a2.legend()
        a2.set_title(f'{title} — Loss')
    plt.tight_layout(); plt.savefig(fname); plt.close()

train_fig(histC, 'Figures/cnn_training.pdf', 'Compact CNN')
train_fig(histD, 'Figures/cnn2_training.pdf', 'Deeper CNN')

# ---- confusion matrix helper ----
def cm_fig(cm, fname, title):
    cm = np.array(cm)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, M, t, fmt in [(a1, cmn, 'Normalized', '.2f'), (a2, cm, 'Absolute', 'd')]:
        im = ax.imshow(M, cmap='Blues', vmin=0, vmax=M.max())
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(SHORT, fontsize=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{title} — {t}'); ax.grid(False)
        thr = M.max()*.55
        for i in range(4):
            for j in range(4):
                v = M[i, j]
                ax.text(j, i, format(v, fmt), ha='center', va='center', fontsize=9,
                        color='white' if v > thr else 'black')
    plt.tight_layout(); plt.savefig(fname); plt.close()

cm_fig(resC['cm'], 'Figures/cnn_confusion.pdf', 'Compact CNN')
cm_fig(resD['cm'], 'Figures/cnn2_confusion.pdf', 'Deeper CNN')

# ---- per-class F1 comparison ----
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(4); w = .38
f1C = [resC['report'][c]['f1-score'] for c in CL]
f1D = [resD['report'][c]['f1-score'] for c in CL]
ax.bar(x - w/2, f1C, w, label='Compact CNN', color=BLUE, edgecolor='black', lw=.5)
ax.bar(x + w/2, f1D, w, label='Deeper CNN', color=GREEN, edgecolor='black', lw=.5)
ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=15)
ax.set_ylabel('F1-score'); ax.set_ylim(0, 1); ax.legend()
ax.set_title('Per-class F1-score comparison')
for i in range(4):
    ax.text(i - w/2, f1C[i]+.02, f'{f1C[i]:.2f}', ha='center', fontsize=8)
    ax.text(i + w/2, f1D[i]+.02, f'{f1D[i]:.2f}', ha='center', fontsize=8)
plt.tight_layout(); plt.savefig('Figures/perclass_f1.pdf'); plt.close()

# ---- ROC one-vs-rest ----
from sklearn.metrics import roc_curve, auc as aucf
fig, ax = plt.subplots(figsize=(7, 6.5))
styles = [('Compact CNN', resC, '-'), ('Deeper CNN', resD, '--')]
colors = [GREEN, ORANGE, BLUE, RED]
for label, res, ls in styles:
    yte = np.array(res['yte']); prob = np.array(res['prob'])
    Y = np.eye(4)[yte]
    for k in range(4):
        fpr, tpr, _ = roc_curve(Y[:, k], prob[:, k])
        a = aucf(fpr, tpr)
        ax.plot(fpr, tpr, ls, color=colors[k], lw=1.6,
                label=f'{label}: {SHORT[k]} (AUC={a:.2f})')
ax.plot([0, 1], [0, 1], 'k--', lw=.8, alpha=.5)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves (One-vs-Rest)')
ax.legend(fontsize=7, loc='lower right')
plt.tight_layout(); plt.savefig('Figures/roc_curves.pdf'); plt.close()

# ---- model comparison bar ----
fig, ax = plt.subplots(figsize=(7.5, 4.5))
models_ = ['Compact CNN', 'Deeper CNN']
accs = [resC['report']['accuracy'], resD['report']['accuracy']]
f1s = [resC['report']['macro avg']['f1-score'], resD['report']['macro avg']['f1-score']]
aucs = [resC['auc'], resD['auc']]
x = np.arange(2); w = .25
ax.bar(x - w, accs, w, label='Test Accuracy', color=BLUE, edgecolor='black', lw=.5)
ax.bar(x, f1s, w, label='Macro F1', color=GREEN, edgecolor='black', lw=.5)
ax.bar(x + w, aucs, w, label='Macro AUC', color=ORANGE, edgecolor='black', lw=.5)
ax.set_xticks(x); ax.set_xticklabels(models_)
ax.set_ylim(0, 1); ax.legend(); ax.set_title('Overall model comparison')
for i in range(2):
    ax.text(i - w, accs[i]+.02, f'{accs[i]:.2f}', ha='center', fontsize=8)
    ax.text(i, f1s[i]+.02, f'{f1s[i]:.2f}', ha='center', fontsize=8)
    ax.text(i + w, aucs[i]+.02, f'{aucs[i]:.2f}', ha='center', fontsize=8)
plt.tight_layout(); plt.savefig('Figures/comparison.pdf'); plt.close()

print('figures done')
for f in sorted(os.listdir('Figures')):
    print(' ', f)
