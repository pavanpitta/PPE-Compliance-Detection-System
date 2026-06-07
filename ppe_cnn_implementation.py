import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

DATASET_ROOT = "./data/ppe_dataset"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_SCRATCH = 50
EPOCHS_TL_HEAD = 20
EPOCHS_TL_FINE = 30
LR = 0.001
FINE_LR = 0.0001

CLASS_NAMES = ["PPE_Compliant", "HardHat_Only", "Vest_Only", "No_PPE"]
NUM_CLASSES = len(CLASS_NAMES)


def load_dataset(root, img_size=IMG_SIZE):
    images, labels, paths = [], [], []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = Path(root) / cls
        if not cls_dir.exists():
            print(f"Missing: {cls_dir}")
            continue
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in valid_exts]
        print(f"{cls}: {len(files)} images")
        for fp in files:
            try:
                img = load_img(str(fp), target_size=(img_size, img_size))
                images.append(img_to_array(img))
                labels.append(idx)
                paths.append(str(fp))
            except Exception as e:
                print(f"Skipped {fp.name}: {e}")

    return np.array(images, dtype='float32'), np.array(labels, dtype='int32'), paths


images_raw, labels, file_paths = load_dataset(DATASET_ROOT)
images_norm = images_raw / 255.0

X_trainval, X_test, y_trainval, y_test = train_test_split(
    images_norm, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15/0.85, stratify=y_trainval, random_state=42
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_gen = train_aug.flow(X_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True, seed=42)
val_gen = ImageDataGenerator().flow(X_val, y_val_cat, batch_size=BATCH_SIZE, shuffle=False)


def plot_class_distribution(labels, save_path):
    counts = pd.Series(labels).map(dict(enumerate(CLASS_NAMES))).value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=['#1A5276', '#2E86C1', '#85C1E9', '#D6EAF8'], edgecolor='white')
    ax.set_title("Class Distribution", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample_images(images, labels, save_path, n=4):
    fig, axes = plt.subplots(NUM_CLASSES, n, figsize=(n*3, NUM_CLASSES*3))
    fig.suptitle("Sample Images per Class", fontsize=14, fontweight='bold')
    for row in range(NUM_CLASSES):
        idxs = np.where(labels == row)[0]
        sample = np.random.choice(idxs, min(n, len(idxs)), replace=False)
        for col, i in enumerate(sample):
            axes[row, col].imshow(images[i])
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(CLASS_NAMES[row], fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_curves(history, name, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training — {name}", fontsize=13, fontweight='bold')
    ax1.plot(history.history['accuracy'], label='Train', color='#1A5276', lw=2)
    ax1.plot(history.history['val_accuracy'], label='Val', color='#E74C3C', lw=2, ls='--')
    ax1.set_title("Accuracy"); ax1.set_ylim(0, 1.05); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history.history['loss'], label='Train', color='#1A5276', lw=2)
    ax2.plot(history.history['val_loss'], label='Val', color='#E74C3C', lw=2, ls='--')
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion(y_true, y_pred, name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {name}", fontsize=13, fontweight='bold')
    for ax, data, title, fmt in zip(axes, [cm, cm_norm], ["Counts", "Normalized"], ['d', '.2f']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
        ax.set_title(title); ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc(y_true, y_score, name, save_path):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    colors = ['#1A5276', '#E74C3C', '#2ECC71', '#F39C12']
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1], [0,1], 'k--', lw=1.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curves — {name}", fontsize=13, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


plot_class_distribution(y_train, f"{OUTPUT_DIR}/01_class_distribution.png")
plot_sample_images(X_train, y_train, f"{OUTPUT_DIR}/02_sample_images.png")


def build_custom_cnn():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs, outputs, name="CustomCNN")
    model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


custom_cnn = build_custom_cnn()
custom_cnn.summary()

cnn_cbs = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(f"{OUTPUT_DIR}/best_custom_cnn.keras", monitor='val_accuracy', save_best_only=True)
]

history_cnn = custom_cnn.fit(
    train_gen,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS_SCRATCH,
    validation_data=val_gen,
    validation_steps=len(X_val) // BATCH_SIZE,
    callbacks=cnn_cbs,
    verbose=1
)

plot_curves(history_cnn, "Custom CNN", f"{OUTPUT_DIR}/03_cnn_curves.png")


def build_mobilenetv2():
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs, outputs, name="MobileNetV2")
    model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base


mobilenet_model, mobilenet_base = build_mobilenetv2()

tl_cbs = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

history_mn1 = mobilenet_model.fit(
    train_gen, steps_per_epoch=len(X_train)//BATCH_SIZE,
    epochs=EPOCHS_TL_HEAD,
    validation_data=val_gen, validation_steps=len(X_val)//BATCH_SIZE,
    callbacks=tl_cbs, verbose=1
)

mobilenet_base.trainable = True
for layer in mobilenet_base.layers:
    if not any(b in layer.name for b in ["block_14", "block_15", "block_16", "Conv_1"]):
        layer.trainable = False

mobilenet_model.compile(optimizer=Adam(FINE_LR), loss='categorical_crossentropy', metrics=['accuracy'])

history_mn2 = mobilenet_model.fit(
    train_gen, steps_per_epoch=len(X_train)//BATCH_SIZE,
    epochs=EPOCHS_TL_FINE,
    validation_data=val_gen, validation_steps=len(X_val)//BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(f"{OUTPUT_DIR}/best_mobilenetv2.keras", monitor='val_accuracy', save_best_only=True)
    ],
    verbose=1
)

class MergedHistory:
    def __init__(self, h1, h2):
        self.history = {k: h1.history[k] + h2.history.get(k, []) for k in h1.history}

history_mobilenet = MergedHistory(history_mn1, history_mn2)
plot_curves(history_mobilenet, "MobileNetV2", f"{OUTPUT_DIR}/04_mobilenet_curves.png")


def build_resnet50():
    base = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs, outputs, name="ResNet50")
    model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base


resnet_model, resnet_base = build_resnet50()

history_rn1 = resnet_model.fit(
    train_gen, steps_per_epoch=len(X_train)//BATCH_SIZE,
    epochs=EPOCHS_TL_HEAD,
    validation_data=val_gen, validation_steps=len(X_val)//BATCH_SIZE,
    callbacks=tl_cbs, verbose=1
)

resnet_base.trainable = True
for layer in resnet_base.layers[:-20]:
    layer.trainable = False

resnet_model.compile(optimizer=Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_rn2 = resnet_model.fit(
    train_gen, steps_per_epoch=len(X_train)//BATCH_SIZE,
    epochs=EPOCHS_TL_FINE,
    validation_data=val_gen, validation_steps=len(X_val)//BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(f"{OUTPUT_DIR}/best_resnet50.keras", monitor='val_accuracy', save_best_only=True)
    ],
    verbose=1
)

history_resnet = MergedHistory(history_rn1, history_rn2)
plot_curves(history_resnet, "ResNet50", f"{OUTPUT_DIR}/05_resnet_curves.png")


def evaluate_model(model, name):
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    _, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
    macro_auc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')

    print(f"\n{name} — Test Accuracy: {acc:.4f} | Macro AUC: {macro_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    tag = name.lower().replace(" ", "_")
    with open(f"{OUTPUT_DIR}/{tag}_report.txt", 'w') as f:
        f.write(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    plot_confusion(y_test, y_pred, name, f"{OUTPUT_DIR}/{tag}_confusion.png")
    plot_roc(y_test, y_prob, name, f"{OUTPUT_DIR}/{tag}_roc.png")

    return {'name': name, 'accuracy': acc, 'auc': macro_auc, 'y_pred': y_pred, 'y_prob': y_prob}


r_cnn = evaluate_model(custom_cnn, "Custom CNN")
r_mn  = evaluate_model(mobilenet_model, "MobileNetV2")
r_rn  = evaluate_model(resnet_model, "ResNet50")

all_results = [r_cnn, r_mn, r_rn]
names = [r['name'] for r in all_results]
accs  = [r['accuracy'] for r in all_results]
aucs  = [r['auc'] for r in all_results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#1A5276', '#2E86C1', '#85C1E9']
for ax, vals, title in zip([ax1, ax2], [accs, aucs], ["Test Accuracy", "Macro AUC"]):
    bars = ax.bar(names, vals, color=colors, edgecolor='white')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.4f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nModel Comparison:")
print(f"{'Model':<20} {'Accuracy':>10} {'AUC':>10}")
for r in all_results:
    print(f"{r['name']:<20} {r['accuracy']:>10.4f} {r['auc']:>10.4f}")


def compute_gradcam(model, img_array, class_idx, last_conv):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    img_u8 = (img * 255).astype(np.uint8)
    h_u8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(cv2.resize(h_u8, (img_u8.shape[1], img_u8.shape[0])), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_u8, 1 - alpha, colored, alpha, 0)


def plot_gradcam_grid(model, name, last_conv, save_path, n=4):
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    correct = np.where(y_pred == y_test)[0]
    wrong   = np.where(y_pred != y_test)[0]
    correct_sample = np.random.choice(correct, min(n, len(correct)), replace=False)
    wrong_sample   = np.random.choice(wrong,   min(n, len(wrong)),   replace=False)

    fig, axes = plt.subplots(2, n, figsize=(n*4, 9))
    fig.suptitle(f"Grad-CAM — {name}", fontsize=14, fontweight='bold')

    for col, idx in enumerate(correct_sample):
        hm = compute_gradcam(model, X_test[idx][np.newaxis], y_pred[idx], last_conv)
        axes[0, col].imshow(overlay_gradcam(X_test[idx], hm))
        axes[0, col].set_title(f"True: {CLASS_NAMES[y_test[idx]]}\nPred: {CLASS_NAMES[y_pred[idx]]}",
                                fontsize=8, color='green', fontweight='bold')
        axes[0, col].axis('off')

    for col, idx in enumerate(wrong_sample):
        hm = compute_gradcam(model, X_test[idx][np.newaxis], y_pred[idx], last_conv)
        axes[1, col].imshow(overlay_gradcam(X_test[idx], hm))
        axes[1, col].set_title(f"True: {CLASS_NAMES[y_test[idx]]}\nPred: {CLASS_NAMES[y_pred[idx]]}",
                                fontsize=8, color='red', fontweight='bold')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("Correct", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Misclassified", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


plot_gradcam_grid(custom_cnn, "Custom CNN", "activation_3", f"{OUTPUT_DIR}/07_gradcam_cnn.png")
plot_gradcam_grid(mobilenet_model, "MobileNetV2", "Conv_1", f"{OUTPUT_DIR}/08_gradcam_mobilenet.png")


def plot_misclassifications(model, name, save_path, n=10):
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_conf = np.max(y_prob, axis=1)
    wrong = np.where(y_pred != y_test)[0]
    top_wrong = wrong[np.argsort(-y_conf[wrong])[:n]]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle(f"Top {n} Confident Misclassifications — {name}", fontsize=13, fontweight='bold')
    for i, idx in enumerate(top_wrong):
        axes[i].imshow(X_test[idx])
        axes[i].set_title(
            f"True: {CLASS_NAMES[y_test[idx]]}\nPred: {CLASS_NAMES[y_pred[idx]]}\n{y_conf[idx]:.2%}",
            fontsize=8, color='red'
        )
        axes[i].axis('off')
    for j in range(len(top_wrong), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


plot_misclassifications(mobilenet_model, "MobileNetV2", f"{OUTPUT_DIR}/09_misclassifications.png")

custom_cnn.save(f"{OUTPUT_DIR}/final_custom_cnn.keras")
mobilenet_model.save(f"{OUTPUT_DIR}/final_mobilenetv2.keras")
resnet_model.save(f"{OUTPUT_DIR}/final_resnet50.keras")

print("\nDone. All outputs saved to:", os.path.abspath(OUTPUT_DIR))
