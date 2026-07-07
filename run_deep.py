import json, numpy as np, os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_NUM_INTRAOP_THREADS'] = '3'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import PIL.Image as Image

np.random.seed(42); tf.random.set_seed(42)
name = 'DeepCNN'; ep = int(sys.argv[1]) if len(sys.argv) > 1 else 45
STAT = f'status_{name}.txt'
def log(s): open(STAT, 'a').write(s + '\n')
open(STAT, 'w').write('start\n')

rows = json.load(open('crop_labels.json'))
CL = ['PPE_Compliant', 'HardHat_Only', 'Vest_Only', 'NO_PPE']
c2i = {c: i for i, c in enumerate(CL)}
IMG = 96  # keep same input size as CustomCNN for fair comparison
paths = [r[0] for r in rows]; ys = [c2i[r[1]] for r in rows]
ptr, ptmp, ytr, ytmp = train_test_split(paths, ys, test_size=.3, stratify=ys, random_state=42)
pva, pte, yva, yte = train_test_split(ptmp, ytmp, test_size=.5, stratify=ytmp, random_state=42)

def load(ps):
    X = np.zeros((len(ps), IMG, IMG, 3), np.float32)
    for i, p in enumerate(ps):
        X[i] = np.asarray(Image.open(p).convert('RGB').resize((IMG, IMG)), np.float32) / 255.
    return X

Xtr, Xva, Xte = load(ptr), load(pva), load(pte)
ytr, yva, yte = np.array(ytr), np.array(yva), np.array(yte)
cw = {i: w for i, w in enumerate(compute_class_weight('balanced', classes=np.arange(4), y=ytr))}
log(f'data loaded {Xtr.shape}')

aug = tf.keras.Sequential([layers.RandomFlip('horizontal'), layers.RandomRotation(.05), layers.RandomZoom(.1)])

def cbr(m, f, s=1):
    m.add(layers.Conv2D(f, 3, strides=s, padding='same', use_bias=False))
    m.add(layers.BatchNormalization()); m.add(layers.Activation('relu'))

# Deeper network but uses strided convs (fewer, cheaper ops than double-conv+pool at 96px)
m = models.Sequential([layers.Input((IMG, IMG, 3)), aug])
cbr(m, 32); cbr(m, 32, s=2)      # 96 -> 48
cbr(m, 64); cbr(m, 64, s=2)      # 48 -> 24
cbr(m, 128); cbr(m, 128, s=2)    # 24 -> 12
cbr(m, 256); cbr(m, 256, s=2)    # 12 -> 6
cbr(m, 512)                      # 6 -> 6
m.add(layers.GlobalAveragePooling2D())
m.add(layers.Dense(256, activation='relu')); m.add(layers.Dropout(.4))
m.add(layers.Dense(128, activation='relu')); m.add(layers.Dropout(.3))
m.add(layers.Dense(4, activation='softmax'))
m.compile(tf.keras.optimizers.Adam(5e-4), 'sparse_categorical_crossentropy', metrics=['accuracy'])

class CB(tf.keras.callbacks.Callback):
    def on_epoch_end(self, e, logs=None):
        log(f"ep{e+1} acc={logs['accuracy']:.3f} val={logs['val_accuracy']:.3f} loss={logs['loss']:.3f} vloss={logs['val_loss']:.3f}")

cbs = [CB(),
       tf.keras.callbacks.ModelCheckpoint(f'best_{name}.keras', monitor='val_accuracy', save_best_only=True),
       tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor='val_accuracy'),
       tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=.5, min_lr=1e-5)]
h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=ep, batch_size=32,
          class_weight=cw, callbacks=cbs, verbose=0)
json.dump({k: [float(x) for x in v] for k, v in h.history.items()}, open(f'hist_{name}.json', 'w'))
log('fit done')
import gc; del Xtr, Xva; gc.collect()
prob = m.predict(Xte, verbose=0, batch_size=16); pred = prob.argmax(1)
rep = classification_report(yte, pred, target_names=CL, output_dict=True, zero_division=0)
auc = roc_auc_score(tf.keras.utils.to_categorical(yte, 4), prob, multi_class='ovr', average='macro')
json.dump({'report': rep, 'cm': confusion_matrix(yte, pred).tolist(), 'auc': float(auc),
           'params': int(m.count_params()), 'prob': prob.tolist(), 'yte': yte.tolist()},
          open(f'res_{name}.json', 'w'))
log(f"DONE acc={rep['accuracy']:.4f} f1={rep['macro avg']['f1-score']:.4f} auc={auc:.4f}")
