import json, numpy as np, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.model_selection import train_test_split

CL = ['PPE_Compliant', 'HardHat_Only', 'Vest_Only', 'NO_PPE']
c2i = {c: i for i, c in enumerate(CL)}
IMG = 96
rows = json.load(open('crop_labels.json'))
paths = [r[0] for r in rows]; ys = [c2i[r[1]] for r in rows]
ptr, ptmp, ytr, ytmp = train_test_split(paths, ys, test_size=.3, stratify=ys, random_state=42)
pva, pte, yva, yte = train_test_split(ptmp, ytmp, test_size=.5, stratify=ytmp, random_state=42)

m = tf.keras.models.load_model('best_CustomCNN.keras')
# Rebuild inference model without augmentation, then locate last conv
inp = tf.keras.Input((IMG, IMG, 3))
x = inp
started = False
for layer in m.layers:
    if isinstance(layer, tf.keras.Sequential):
        continue  # skip augmentation block (inference: identity)
    x = layer(x)
    if layer.name == 'conv2d_3':
        conv_out = x
infer = tf.keras.Model(inp, x)
grad_model = tf.keras.models.Model(inp, [conv_out, x])

def load1(p):
    return np.asarray(Image.open(p).convert('RGB').resize((IMG, IMG)), np.float32) / 255.

def gradcam(x, cls):
    xt = tf.convert_to_tensor(x[None])
    with tf.GradientTape() as tape:
        conv, pred = grad_model(xt)
        loss = pred[:, cls]
    grads = tape.gradient(loss, conv)
    w = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv[0] * w, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((IMG, IMG))) / 255.
    return cam

# predict test set
Xte = np.stack([load1(p) for p in pte])
prob = infer.predict(Xte, verbose=0, batch_size=16)
pred = prob.argmax(1); yte = np.array(yte)

# pick correct + wrong examples per class
correct, wrong = {}, {}
for i in range(len(pte)):
    t = yte[i]
    if pred[i] == t and t not in correct:
        correct[t] = i
    if pred[i] != t and t not in wrong:
        wrong[t] = i

fig, axes = plt.subplots(4, 4, figsize=(11, 11))
for col, c in enumerate(CL):
    for row, (grp, tag) in enumerate([(correct, 'Correct'), (wrong, 'Wrong')]):
        idx_orig = row*2
        if c2i[c] in grp:
            i = grp[c2i[c]]
            x = Xte[i]; cam = gradcam(x, pred[i])
            axes[idx_orig, col].imshow(x)
            axes[idx_orig, col].set_title(f'{c}\n({tag}) orig', fontsize=8)
            axes[idx_orig+1, col].imshow(x)
            axes[idx_orig+1, col].imshow(cam, cmap='jet', alpha=.5)
            axes[idx_orig+1, col].set_title(f'pred: {CL[pred[i]]}', fontsize=8)
        for r in [idx_orig, idx_orig+1]:
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([]); axes[r, col].grid(False)
plt.suptitle('Grad-CAM: original (rows 1,3) and heatmap overlay (rows 2,4) — Compact CNN',
             fontweight='bold', fontsize=12)
plt.tight_layout(); plt.savefig('Figures/gradcam.pdf'); plt.close()
print('gradcam done')
