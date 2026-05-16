"""
models/train_model.py
=====================
Plant Disease Detection — EfficientNetB0 Transfer Learning
----------------------------------------------------------
Upgraded from MobileNetV2 → EfficientNetB0 for better accuracy.

Dataset  : PlantVillage
Classes  : 9 classes (Tomato, Potato, Pepper)
Output   : models/saved_model/plant_disease_model.h5
           models/saved_model/class_labels.json

HOW TO RUN
----------
1. Make sure your dataset is at:
   data/plantvillage/PlantVillage/
       Tomato_healthy/
       Tomato_Early_blight/
       Potato___healthy/
       ... etc

2. Run from the agri_edge/ root folder:
   python models/train_model.py

3. Expected runtime on CPU laptop: ~40–55 minutes
"""

import os
import sys
import json
import shutil
import random
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0          # ← upgraded from MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"[INFO] TensorFlow version : {tf.__version__}")
print(f"[INFO] GPUs available     : {len(tf.config.list_physical_devices('GPU'))}")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Dataset path ──────────────────────────────────────────────────────────────
# If your folders are inside a PlantVillage subfolder, use:
#   DATASET_ROOT = "data/plantvillage/PlantVillage"
# If they are directly inside data/plantvillage, use:
#   DATASET_ROOT = "data/plantvillage"
DATASET_ROOT = "data/plantvillage/PlantVillage"

# ── 9 classes — all diseases available in the dataset ─────────────────────────
# Left side  = exact folder name in your dataset (must match perfectly)
# Right side = display label shown in the app
TARGET_CLASSES = {
    "Pepper__bell___Bacterial_spot" : "Pepper - Bacterial Spot",
    "Pepper__bell___healthy"        : "Pepper - Healthy",
    "Potato___Early_blight"         : "Potato - Early Blight",
    "Potato___healthy"              : "Potato - Healthy",
    "Potato___Late_blight"          : "Potato - Late Blight",
    "Tomato_Bacterial_spot"         : "Tomato - Bacterial Spot",
    "Tomato_Early_blight"           : "Tomato - Early Blight",
    "Tomato_healthy"                : "Tomato - Healthy",
    "Tomato_Late_blight"            : "Tomato - Late Blight",
}

# ── Training settings ─────────────────────────────────────────────────────────
IMG_SIZE             = (224, 224)   # EfficientNetB0 works well at 224×224
BATCH_SIZE           = 16           # Keep low for laptop RAM
EPOCHS_HEAD          = 15           # Phase 1: train head only (was 10)
EPOCHS_FINETUNE      = 10           # Phase 2: fine-tune top layers (was 5)
VALIDATION_SPLIT     = 0.2          # 80% train / 20% validation
MAX_IMAGES_PER_CLASS = 800          # More images = better accuracy (was 500)
FINETUNE_LAYERS      = 30           # Unfreeze more layers for fine-tuning (was 20)

# ── Output paths ──────────────────────────────────────────────────────────────
SAVE_DIR         = "models/saved_model"
MODEL_SAVE_PATH  = os.path.join(SAVE_DIR, "plant_disease_model.h5")
LABELS_SAVE_PATH = os.path.join(SAVE_DIR, "class_labels.json")
PLOTS_SAVE_PATH  = os.path.join(SAVE_DIR, "training_curves.png")
STAGING_DIR      = "data/plantvillage_subset"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DATASET PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset():
    """Check that DATASET_ROOT exists and all target class folders are present."""
    if not os.path.isdir(DATASET_ROOT):
        print(f"\n❌  Dataset not found at: '{DATASET_ROOT}'")
        print("    Check your DATASET_ROOT path at the top of this file.")
        sys.exit(1)

    available = os.listdir(DATASET_ROOT)
    missing   = [c for c in TARGET_CLASSES if c not in available]

    if missing:
        print(f"\n⚠️  Some class folders not found in '{DATASET_ROOT}':")
        for m in missing:
            print(f"    ✗ {m}")
        print(f"\n    Available folders: {available[:15]}")
        print("\n    Fix: update TARGET_CLASSES or DATASET_ROOT to match your folder names.\n")
        sys.exit(1)

    print(f"[INFO] Dataset validated — {len(TARGET_CLASSES)} classes found at: {DATASET_ROOT}")


def build_subset():
    """
    Copy only the target class folders into a clean staging directory.
    Caps images per class at MAX_IMAGES_PER_CLASS.
    """
    if os.path.isdir(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    print(f"\n[STEP 1] Building {len(TARGET_CLASSES)}-class subset → {STAGING_DIR}")

    counts = {}
    for folder_name, display_label in TARGET_CLASSES.items():
        src  = os.path.join(DATASET_ROOT, folder_name)
        dest = os.path.join(STAGING_DIR, folder_name)
        os.makedirs(dest, exist_ok=True)

        images = [
            f for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) > MAX_IMAGES_PER_CLASS:
            images = random.sample(images, MAX_IMAGES_PER_CLASS)

        for img in images:
            shutil.copy(os.path.join(src, img), os.path.join(dest, img))

        counts[display_label] = len(images)
        print(f"    ✓  {display_label:<35}  {len(images):>4} images")

    print(f"\n    Total images in subset: {sum(counts.values())}")
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DATA GENERATORS  (stronger augmentation)
# ══════════════════════════════════════════════════════════════════════════════

def create_generators():
    """
    Build train and validation data generators.

    Key upgrade: EfficientNet uses its own preprocessing (scales 0–255 → 0–1
    internally), so we do NOT divide by 255 ourselves.
    Augmentation is significantly stronger than before for better generalisation.
    """
    print(f"\n[STEP 2] Creating data generators  (batch_size={BATCH_SIZE})")

    # EfficientNet preprocessing — do NOT use mobilenet_v2.preprocess_input here
    preprocess = tf.keras.applications.efficientnet.preprocess_input

    # Training set — heavy augmentation to handle real-world leaf photos
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=40,           # Leaves appear at any angle in the field
        width_shift_range=0.20,      # Horizontal position variation
        height_shift_range=0.20,     # Vertical position variation
        shear_range=0.15,            # Slight perspective distortion
        zoom_range=0.25,             # Camera distance variation
        horizontal_flip=True,        # Leaves can face either direction
        vertical_flip=True,          # Also flip vertically — helps generalise
        brightness_range=[0.6, 1.4], # Handle dark/bright/cloudy lighting
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )

    # Validation set — ONLY preprocessing, zero augmentation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        STAGING_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        STAGING_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    print(f"    Train samples      : {train_gen.samples}")
    print(f"    Validation samples : {val_gen.samples}")
    print(f"    Classes found      : {train_gen.class_indices}")
    return train_gen, val_gen


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — MODEL ARCHITECTURE  (EfficientNetB0)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int):
    """
    EfficientNetB0 transfer learning model.

    Why EfficientNetB0 over MobileNetV2?
    - Scales width, depth, AND resolution together (compound scaling)
    - ~5-10% better accuracy on image classification tasks
    - Similar inference speed on CPU
    - Works great for plant disease detection

    Architecture:
        Input (224×224×3)
            ↓
        EfficientNetB0 (ImageNet weights, FROZEN in Phase 1)
            ↓
        GlobalAveragePooling2D
            ↓
        BatchNormalization
            ↓
        Dense(256, relu)       ← larger head than before (was 128)
            ↓
        Dropout(0.4)           ← slightly higher dropout (was 0.3)
            ↓
        Dense(num_classes, softmax)
    """
    print(f"\n[STEP 3] Building EfficientNetB0 model  ({num_classes} classes)")

    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False   # Freeze all base layers for Phase 1

    # Functional API
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3), name="leaf_image")
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.BatchNormalization(name="bn")(x)
    x   = layers.Dense(256, activation="relu", name="fc256")(x)   # larger than before
    x   = layers.Dropout(0.4, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inp, out, name="AgriEdge_EfficientNetB0")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    total     = model.count_params()
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"    Total params     : {total:,}")
    print(f"    Trainable params : {trainable:,}  (head only — base frozen)")

    return model, base


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def get_callbacks(phase: int) -> list:
    """
    Phase 1: ModelCheckpoint + EarlyStopping + ReduceLROnPlateau
    Phase 2: ModelCheckpoint + EarlyStopping (tighter patience)
    """
    cb = []

    cb.append(callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max",
    ))

    cb.append(callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5 if phase == 1 else 4,   # slightly more patient than before
        restore_best_weights=True,
        verbose=1,
    ))

    if phase == 1:
        cb.append(callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ))

    return cb


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — PHASE 1: TRAIN HEAD
# ══════════════════════════════════════════════════════════════════════════════

def phase1_train_head(model, train_gen, val_gen):
    """Train only the classification head. EfficientNetB0 base stays frozen."""
    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Head Training")
    print(f"  Epochs: up to {EPOCHS_HEAD}  |  LR: 1e-3  |  Base: FROZEN")
    print(f"{'='*60}\n")

    t0 = time.time()
    history = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=get_callbacks(phase=1),
        verbose=1,
    )
    elapsed = time.time() - t0

    best = max(history.history["val_accuracy"])
    print(f"\n  [Phase 1 done]  best val_accuracy={best*100:.2f}%  time={elapsed/60:.1f} min")
    return history


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — PHASE 2: FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

def phase2_finetune(model, base, train_gen, val_gen):
    """
    Unfreeze the last FINETUNE_LAYERS of EfficientNetB0 and train at LR=1e-5.
    The very low learning rate prevents destroying pretrained weights.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Fine-Tuning")
    print(f"  Epochs: up to {EPOCHS_FINETUNE}  |  LR: 1e-5  |  Top {FINETUNE_LAYERS} layers unfrozen")
    print(f"{'='*60}\n")

    base.trainable = True
    n            = len(base.layers)
    freeze_until = n - FINETUNE_LAYERS

    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= freeze_until)

    unfrozen = sum(1 for l in base.layers if l.trainable)
    print(f"  EfficientNetB0 layers : {n} total  |  frozen: {n - unfrozen}  |  unfrozen: {unfrozen}")

    # Must recompile after changing trainability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    t0 = time.time()
    history = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=get_callbacks(phase=2),
        verbose=1,
    )
    elapsed = time.time() - t0

    best = max(history.history["val_accuracy"])
    print(f"\n  [Phase 2 done]  best val_accuracy={best*100:.2f}%  time={elapsed/60:.1f} min")
    return history


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, val_gen, class_indices: dict) -> float:
    """Run full evaluation with per-class accuracy and confusion matrix."""
    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION  (validation set)")
    print(f"{'='*60}")

    val_gen.reset()

    all_preds, all_true = [], []
    for i in range(len(val_gen)):
        xb, yb = val_gen[i]
        preds   = model.predict(xb, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_true.extend(np.argmax(yb, axis=1))

    all_preds  = np.array(all_preds)
    all_true   = np.array(all_true)
    idx2folder = {v: k for k, v in class_indices.items()}
    n          = len(class_indices)

    overall = np.mean(all_preds == all_true)
    print(f"\n  Overall Accuracy : {overall*100:.2f}%\n")

    print(f"  {'Class':<35} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'-'*60}")
    for idx in range(n):
        mask    = all_true == idx
        correct = np.sum(all_preds[mask] == idx)
        total   = np.sum(mask)
        acc     = correct / total if total else 0
        label   = TARGET_CLASSES.get(idx2folder.get(idx, ""), f"Class {idx}")
        print(f"  {label:<35} {correct:>8} {total:>8} {acc*100:>7.1f}%")

    return float(overall)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_curves(h1, h2):
    """Save a 2×2 grid of accuracy / loss / precision / recall curves."""

    def join(key):
        return h1.history.get(key, []) + h2.history.get(key, [])

    ep1   = len(h1.history["accuracy"])
    total = ep1 + len(h2.history["accuracy"])
    xs    = range(1, total + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Agri-Edge — EfficientNetB0 Plant Disease Training", fontsize=14, fontweight="bold")

    plots = [
        ("accuracy",  "val_accuracy",  "Accuracy",  axes[0, 0]),
        ("loss",      "val_loss",      "Loss",       axes[0, 1]),
        ("precision", "val_precision", "Precision",  axes[1, 0]),
        ("recall",    "val_recall",    "Recall",     axes[1, 1]),
    ]

    for train_key, val_key, title, ax in plots:
        tv = join(train_key)
        vv = join(val_key)
        ax.plot(list(xs)[:len(tv)], tv, "b-o", ms=3, label="Train")
        ax.plot(list(xs)[:len(vv)], vv, "r-o", ms=3, label="Val")
        ax.axvline(ep1 + 0.5, color="gray", ls="--", alpha=0.5, label="Phase 2 start")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if "accuracy" in train_key or title in ("Precision", "Recall"):
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(PLOTS_SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved → {PLOTS_SAVE_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 — SAVE LABELS
# ══════════════════════════════════════════════════════════════════════════════

def save_labels(class_indices: dict):
    """
    Write class_labels.json — index position matches model output neuron.
    disease_detector.py reads this file at inference time.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    ordered = sorted(class_indices.items(), key=lambda x: x[1])
    labels  = [TARGET_CLASSES.get(folder, folder) for folder, _ in ordered]

    with open(LABELS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"[INFO] Labels saved → {LABELS_SAVE_PATH}")
    for i, lbl in enumerate(labels):
        print(f"         [{i}] {lbl}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10 — UPDATE DISEASE_DETECTOR PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def update_detector_preprocessing():
    """
    Patch disease_detector.py to use EfficientNet preprocessing
    instead of MobileNetV2 preprocessing.
    This ensures the inference pipeline matches training exactly.
    """
    detector_path = "models/disease_detector.py"
    if not os.path.exists(detector_path):
        print(f"[WARN] Could not find {detector_path} — skipping auto-patch.")
        return

    with open(detector_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace MobileNetV2 preprocessing with EfficientNet preprocessing
    old = "img_array = (img_array / 127.5) - 1.0"
    new = "img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)"

    if old in content:
        # Need to ensure tf is imported in disease_detector
        if "import tensorflow as tf" not in content:
            content = content.replace(
                "import numpy as np",
                "import numpy as np\nimport tensorflow as tf",
            )
        content = content.replace(old, new)
        with open(detector_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[INFO] Auto-patched {detector_path} → EfficientNet preprocessing")
    elif new in content:
        print(f"[INFO] {detector_path} already uses EfficientNet preprocessing")
    else:
        print(f"[WARN] Could not auto-patch {detector_path}.")
        print(f"       Manually replace this line in preprocess_image():")
        print(f"         OLD: img_array = (img_array / 127.5) - 1.0")
        print(f"         NEW: img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    random.seed(42)

    print("\n" + "="*60)
    print("  🌿  Agri-Edge — EfficientNetB0 Disease Classifier Training")
    print("="*60)
    print(f"  Classes  : {len(TARGET_CLASSES)}")
    print(f"  Images   : up to {MAX_IMAGES_PER_CLASS} per class")
    print(f"  Epochs   : {EPOCHS_HEAD} (head) + {EPOCHS_FINETUNE} (fine-tune)")
    print(f"  Dataset  : {DATASET_ROOT}")

    # 1. Validate & stage data
    validate_dataset()
    build_subset()

    # 2. Data generators
    train_gen, val_gen = create_generators()
    num_classes = len(train_gen.class_indices)

    # 3. Build model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model, base = build_model(num_classes)

    # 4. Phase 1 — train head
    h1 = phase1_train_head(model, train_gen, val_gen)

    # 5. Phase 2 — fine-tune
    h2 = phase2_finetune(model, base, train_gen, val_gen)

    # 6. Load best checkpoint and evaluate
    print(f"\n[INFO] Loading best checkpoint: {MODEL_SAVE_PATH}")
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    acc = evaluate(best_model, val_gen, train_gen.class_indices)

    # 7. Save training curves
    plot_curves(h1, h2)

    # 8. Save class labels
    save_labels(train_gen.class_indices)

    # 9. Auto-patch disease_detector.py preprocessing
    update_detector_preprocessing()

    # 10. Summary
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  ✅  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Final val accuracy : {acc*100:.2f}%")
    print(f"  Total time         : {elapsed:.1f} minutes")
    print(f"  Model saved        : {MODEL_SAVE_PATH}")
    print(f"  Labels saved       : {LABELS_SAVE_PATH}")
    print(f"  Curves saved       : {PLOTS_SAVE_PATH}")
    print(f"\n  Next step → python -m streamlit run app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()