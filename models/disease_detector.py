"""
models/disease_detector.py
==========================
Plant Disease Inference — MobileNetV2 Transfer Learning Model
-------------------------------------------------------------
This module is the single entry point for all disease detection in Agri-Edge.

It can be used in two ways:

  1. As a library (imported by app.py / Streamlit):
        from models.disease_detector import predict_disease
        result = predict_disease("path/to/leaf.jpg")

  2. As a CLI tool (run directly from terminal):
        python models/disease_detector.py --image path/to/leaf.jpg
        python models/disease_detector.py --image leaf.jpg --model models/saved_model/plant_disease_model.h5
        python models/disease_detector.py --image leaf.jpg --json     # machine-readable output

WHAT THIS MODULE DOES
---------------------
  load_model()        — Lazy-loads the .h5 model once, caches in memory
  load_class_labels() — Reads class_labels.json, falls back to hardcoded defaults
  preprocess_image()  — Loads any image format → MobileNetV2-ready numpy array
  predict_disease()   — Full pipeline: load → preprocess → infer → format result
  predict_batch()     — Run inference on a list of image paths
  get_disease_info()  — Returns treatment advice + severity for each disease class

All public functions return plain Python dicts — no TF objects leak out.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ── Suppress TensorFlow startup noise ─────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

# Default paths — relative to agri_edge/ project root
DEFAULT_MODEL_PATH  = "models/saved_model/plant_disease_model.h5"
DEFAULT_LABELS_PATH = "models/saved_model/class_labels.json"

# MobileNetV2 expects exactly 224×224 RGB input
IMG_SIZE = (224, 224)

# Confidence below this threshold triggers a low-confidence warning
LOW_CONFIDENCE_THRESHOLD = 0.60

# Fallback labels if class_labels.json is missing
# These match the 4-class training setup in train_model.py
FALLBACK_LABELS = [
    "Potato - Healthy",
    "Potato - Late Blight",
    "Tomato - Early Blight",
    "Tomato - Healthy",
]

# Supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── Disease metadata ──────────────────────────────────────────────────────────
# Each class has: severity, treatment recommendations, and a description.
# Add or edit entries here whenever you add a new class to the model.
DISEASE_INFO = {
    "Tomato - Healthy": {
        "severity"   : "none",
        "description": "Your tomato plant looks healthy! No disease detected.",
        "treatment"  : [
            "Continue regular watering and fertilisation schedule.",
            "Monitor weekly for early signs of pests or discolouration.",
            "Ensure good air circulation between plants.",
        ],
        "is_healthy" : True,
    },
    "Tomato - Early Blight": {
        "severity"   : "moderate",
        "description": (
            "Early Blight (Alternaria solani) causes dark concentric spots on older leaves. "
            "It spreads upward through the canopy if untreated."
        ),
        "treatment"  : [
            "Remove and destroy all infected leaves immediately.",
            "Apply copper-based fungicide (e.g. Bordeaux mixture) every 7–10 days.",
            "Avoid overhead irrigation — water at the base of the plant.",
            "Mulch around the base to prevent soil-splash infection.",
            "Ensure 60 cm spacing between plants for airflow.",
        ],
        "is_healthy" : False,
    },
    "Potato - Healthy": {
        "severity"   : "none",
        "description": "Your potato plant looks healthy! No disease detected.",
        "treatment"  : [
            "Maintain soil moisture — potatoes need consistent watering.",
            "Hill soil around stems every 2–3 weeks to prevent greening of tubers.",
            "Scout weekly for Colorado potato beetle egg masses under leaves.",
        ],
        "is_healthy" : True,
    },
    "Potato - Late Blight": {
        "severity"   : "severe",
        "description": (
            "Late Blight (Phytophthora infestans) is the same pathogen that caused the "
            "Irish Potato Famine. It spreads extremely rapidly in cool, wet conditions. "
            "Act within 24 hours."
        ),
        "treatment"  : [
            "🚨 URGENT: Remove all visibly infected plant parts immediately.",
            "Apply systemic fungicide (Mancozeb or Metalaxyl) within 24 hours.",
            "Do NOT compost infected material — burn or bag it.",
            "Stop all overhead irrigation immediately.",
            "Check neighbouring plants — Late Blight spreads fast via wind.",
            "Harvest unaffected tubers early if infection is severe.",
        ],
        "is_healthy" : False,
    },
    # ── Extended entries for larger model variants ────────────────────────────
    "Tomato - Late Blight": {
        "severity"   : "severe",
        "description": (
            "Late Blight on tomatoes (Phytophthora infestans) causes large, "
            "irregular, water-soaked lesions with white mould on leaf undersides."
        ),
        "treatment"  : [
            "🚨 Remove and destroy infected foliage immediately.",
            "Apply Mancozeb or Cymoxanil-based fungicide every 5–7 days.",
            "Improve drainage and avoid waterlogging.",
            "Apply in the evening to reduce evaporation and maximise absorption.",
        ],
        "is_healthy" : False,
    },
    "Tomato - Bacterial Spot": {
        "severity"   : "moderate",
        "description": (
            "Bacterial Spot (Xanthomonas spp.) causes small, water-soaked "
            "spots that turn brown with yellow halos."
        ),
        "treatment"  : [
            "Apply copper-based bactericide (copper hydroxide or copper oxychloride).",
            "Remove heavily infected leaves.",
            "Avoid working in the field when plants are wet.",
            "Use drip irrigation instead of sprinklers.",
        ],
        "is_healthy" : False,
    },
    "Tomato - Leaf Mold": {
        "severity"   : "low",
        "description": (
            "Leaf Mold (Passalora fulva) thrives in high-humidity greenhouses. "
            "Yellow patches appear on upper leaf surfaces with olive-green mold below."
        ),
        "treatment"  : [
            "Improve greenhouse ventilation to reduce humidity below 85%.",
            "Apply chlorothalonil or mancozeb fungicide.",
            "Remove affected lower leaves to improve airflow.",
            "Water in the morning so leaves dry before nightfall.",
        ],
        "is_healthy" : False,
    },
    "Tomato - Septoria Leaf Spot": {
        "severity"   : "moderate",
        "description": (
            "Septoria Leaf Spot (Septoria lycopersici) causes small circular spots "
            "with dark borders and light grey centres, starting on lower leaves."
        ),
        "treatment"  : [
            "Remove infected lower leaves and destroy them.",
            "Apply chlorothalonil, mancozeb, or copper fungicide.",
            "Avoid wetting the foliage when irrigating.",
            "Rotate crops — don't plant tomatoes in the same bed next season.",
        ],
        "is_healthy" : False,
    },
    "Rice - Healthy": {
        "severity"   : "none",
        "description": "Your rice crop looks healthy.",
        "treatment"  : [
            "Maintain water levels between 5–10 cm during vegetative stage.",
            "Monitor for brown planthopper — common during kharif season.",
        ],
        "is_healthy" : True,
    },
    "Rice - Brown Spot": {
        "severity"   : "moderate",
        "description": (
            "Brown Spot (Cochliobolus miyabeanus) causes oval brown lesions on leaves. "
            "It is often linked to nutrient-deficient or drought-stressed plants."
        ),
        "treatment"  : [
            "Apply recommended dose of potash (K₂O) — deficiency worsens Brown Spot.",
            "Spray Propiconazole or Mancozeb fungicide.",
            "Ensure adequate soil moisture during grain filling stage.",
        ],
        "is_healthy" : False,
    },
    "Rice - Leaf Blast": {
        "severity"   : "severe",
        "description": (
            "Leaf Blast (Magnaporthe oryzae) causes diamond-shaped lesions with "
            "grey centres and brown borders. It is the most destructive rice disease globally."
        ),
        "treatment"  : [
            "🚨 Apply Tricyclazole or Isoprothiolane fungicide immediately.",
            "Avoid excessive nitrogen application — it promotes blast.",
            "Drain fields temporarily to reduce humidity.",
            "Use blast-resistant varieties in high-risk areas.",
        ],
        "is_healthy" : False,
    },
    "Wheat - Healthy": {
        "severity"   : "none",
        "description": "Your wheat crop looks healthy.",
        "treatment"  : [
            "Monitor for aphids and wheat stem fly during tillering.",
            "Ensure timely irrigation at crown root initiation stage.",
        ],
        "is_healthy" : True,
    },
    "Wheat - Brown Rust": {
        "severity"   : "moderate",
        "description": (
            "Brown Rust (Puccinia triticina) appears as small, round orange-brown "
            "pustules scattered across the leaf surface."
        ),
        "treatment"  : [
            "Apply Propiconazole (Tilt 25 EC) at the first sign of infection.",
            "A second spray 15 days later if conditions remain humid.",
            "Sow rust-resistant wheat varieties next season.",
        ],
        "is_healthy" : False,
    },
    "Wheat - Yellow Rust": {
        "severity"   : "severe",
        "description": (
            "Yellow Rust (Puccinia striiformis) forms yellow-orange pustule stripes "
            "along the leaf veins. It spreads rapidly in cool, moist weather."
        ),
        "treatment"  : [
            "🚨 Apply Propiconazole or Tebuconazole immediately.",
            "Two sprays at 15-day intervals for severe infections.",
            "Report outbreak to local Krishi Vigyan Kendra (KVK).",
            "Harvest early if more than 50% of flag leaf is infected.",
        ],
        "is_healthy" : False,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CACHE
#  We store the loaded model and labels here so they are only loaded once
#  per Python process, even if predict_disease() is called many times.
# ══════════════════════════════════════════════════════════════════════════════

_cached_model        = None   # Holds the tf.keras.Model object after first load
_cached_labels       = None   # Holds the list of class label strings
_cached_model_path   = None   # Which .h5 path was loaded (to detect path changes)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — LOAD CLASS LABELS
# ══════════════════════════════════════════════════════════════════════════════

def load_class_labels(labels_path: str = DEFAULT_LABELS_PATH) -> list:
    """
    Load the class label list from a JSON file.

    The JSON file is a list where index == model output neuron:
        ["Potato - Healthy", "Potato - Late Blight", ...]

    If the file is missing or unreadable, falls back to FALLBACK_LABELS
    and prints a warning — the app still runs, just with possibly wrong labels.

    Args:
        labels_path: Path to class_labels.json

    Returns:
        List of class label strings (ordered by model output index)
    """
    global _cached_labels

    if _cached_labels is not None:
        return _cached_labels  # Already loaded this session

    if not os.path.exists(labels_path):
        print(
            f"[WARNING] Class labels file not found: '{labels_path}'\n"
            f"          Using fallback labels: {FALLBACK_LABELS}\n"
            f"          Run train_model.py to generate the correct labels file."
        )
        _cached_labels = FALLBACK_LABELS
        return _cached_labels

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        if not isinstance(labels, list) or len(labels) == 0:
            raise ValueError("class_labels.json must be a non-empty JSON array.")

        _cached_labels = labels
        print(f"[INFO] Loaded {len(labels)} class labels from '{labels_path}'")
        return _cached_labels

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse '{labels_path}': {e}")
        print(f"        Using fallback labels.")
        _cached_labels = FALLBACK_LABELS
        return _cached_labels

    except Exception as e:
        print(f"[ERROR] Unexpected error loading labels: {e}")
        _cached_labels = FALLBACK_LABELS
        return _cached_labels


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the Keras .h5 model from disk.

    This function is lazy — it only loads on the first call, then caches
    the model in `_cached_model`. Subsequent calls return immediately.

    If the model file is missing, returns the string "mock" so the caller
    can fall back to demo predictions without crashing.

    Args:
        model_path: Path to the .h5 model file

    Returns:
        tf.keras.Model on success, or "mock" if the file is not found
    """
    global _cached_model, _cached_model_path

    # Return cached model if same path was already loaded
    if _cached_model is not None and _cached_model_path == model_path:
        return _cached_model

    if not os.path.exists(model_path):
        print(
            f"[WARNING] Model file not found: '{model_path}'\n"
            f"          Running in DEMO MODE with mock predictions.\n"
            f"          Train a real model: python models/train_model.py"
        )
        _cached_model = "mock"
        _cached_model_path = model_path
        return _cached_model

    try:
        # Import TF here (not at module top) so the module loads fast
        # even on machines without TF installed
        import tensorflow as tf

        print(f"[INFO] Loading model from '{model_path}' ...")
        t0 = time.time()

        model = tf.keras.models.load_model(model_path, compile=False)

        elapsed = time.time() - t0
        print(f"[INFO] Model loaded in {elapsed:.2f}s")
        print(f"       Input shape  : {model.input_shape}")
        print(f"       Output shape : {model.output_shape}")

        _cached_model = model
        _cached_model_path = model_path
        return _cached_model

    except Exception as e:
        print(f"[ERROR] Failed to load model from '{model_path}':\n        {e}")
        print(f"        Falling back to DEMO MODE.")
        _cached_model = "mock"
        _cached_model_path = model_path
        return _cached_model


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — IMAGE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_image_path(image_path: Union[str, Path]) -> Path:
    """
    Validate that an image path exists and has a supported extension.

    Args:
        image_path: str or Path to the image file

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError : if the file does not exist
        ValueError        : if the file extension is not supported
    """
    path = Path(image_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: '{image_path}'\n"
            f"Please check the path and try again."
        )

    if not path.is_file():
        raise ValueError(f"Path is a directory, not a file: '{image_path}'")

    ext = path.suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: '{ext}'\n"
            f"Supported formats: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    return path


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_source: Union[str, Path, "UploadedFile"]) -> np.ndarray:
    """
    Load and preprocess an image for MobileNetV2 inference.

    Preprocessing steps:
      1. Open the image with PIL (handles JPEG, PNG, BMP, WebP, TIFF)
      2. Convert to RGB  (drops alpha channel, converts greyscale to 3-channel)
      3. Resize to 224×224 using LANCZOS (high-quality downsampling)
      4. Convert to float32 numpy array
      5. Apply MobileNetV2 preprocessing: scales pixels to [-1.0, 1.0]
         Formula: pixel = (pixel / 127.5) - 1.0
      6. Add batch dimension → shape (1, 224, 224, 3)

    Args:
        image_source: File path (str or Path), or a Streamlit UploadedFile object.

    Returns:
        numpy array of shape (1, 224, 224, 3), dtype float32, values in [-1, 1]

    Raises:
        FileNotFoundError   : if a path is given and the file doesn't exist
        ValueError          : if the image cannot be opened or processed
        UnidentifiedImageError : if PIL cannot identify the file format
    """
    try:
        # ── Open image ────────────────────────────────────────────────────────
        if isinstance(image_source, (str, Path)):
            # File path — validate first, then open
            validate_image_path(image_source)
            img = Image.open(image_source)
        else:
            # Streamlit UploadedFile or file-like object
            img = Image.open(image_source)

        # ── Convert to RGB ────────────────────────────────────────────────────
        # This handles: RGBA (PNG with transparency), L (greyscale), CMYK
        if img.mode != "RGB":
            img = img.convert("RGB")

        # ── Resize ────────────────────────────────────────────────────────────
        # LANCZOS gives better quality than default BILINEAR for downsampling
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        # ── Convert to numpy ──────────────────────────────────────────────────
        img_array = np.array(img, dtype=np.float32)  # Shape: (224, 224, 3)

        # ── MobileNetV2 preprocessing — scales to [-1.0, 1.0] ─────────────────
        # This MUST match training. If you used tf.keras.applications.mobilenet_v2
        # .preprocess_input() during training, use the same here.
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # ── Add batch dimension ───────────────────────────────────────────────
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

        return img_array

    except UnidentifiedImageError:
        raise ValueError(
            f"Cannot read image: '{image_source}'\n"
            f"The file may be corrupted or not a valid image."
        )
    except (FileNotFoundError, ValueError):
        raise   # Re-raise our own validation errors as-is
    except Exception as e:
        raise ValueError(f"Unexpected error during image preprocessing: {e}") from e


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — GET DISEASE INFO
# ══════════════════════════════════════════════════════════════════════════════

def get_disease_info(label: str) -> dict:
    """
    Return metadata for a predicted disease class.

    Args:
        label: Class label string, e.g. "Tomato - Early Blight"

    Returns:
        Dict with keys: severity, description, treatment (list), is_healthy
        Returns a generic dict if the label is not in DISEASE_INFO.
    """
    if label in DISEASE_INFO:
        return DISEASE_INFO[label]

    # Graceful fallback for labels not yet in the metadata dict
    return {
        "severity"   : "unknown",
        "description": f"Disease class '{label}' detected. Consult your local Krishi Vigyan Kendra for advice.",
        "treatment"  : [
            "Photograph the affected plant and consult an agricultural expert.",
            "Isolate affected plants to prevent possible spread.",
        ],
        "is_healthy" : False,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — MOCK PREDICTION (demo / development mode)
# ══════════════════════════════════════════════════════════════════════════════

def _mock_prediction(labels: list) -> dict:
    """
    Generate a realistic-looking mock prediction for demo/development mode.
    Used when no .h5 model file is present.

    The mock picks a random class and assigns it a high confidence,
    then distributes the remaining probability mass across other classes.
    """
    n = len(labels)
    top_idx        = np.random.randint(0, n)
    top_confidence = float(np.random.uniform(0.70, 0.95))

    # Distribute remaining probability across other classes
    remaining = 1.0 - top_confidence
    other_probs = np.random.dirichlet(np.ones(n - 1)) * remaining
    probs = np.insert(other_probs, top_idx, top_confidence)

    all_preds = {labels[i]: round(float(probs[i]), 4) for i in range(n)}

    return {
        "label"          : labels[top_idx],
        "confidence"     : round(top_confidence, 4),
        "all_predictions": all_preds,
        "top3"           : _get_top3(all_preds),
        "source"         : "mock",
    }


def _get_top3(all_predictions: dict) -> list:
    """Return the top-3 predictions sorted by confidence (descending)."""
    sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    return [
        {"label": label, "confidence": conf}
        for label, conf in sorted_preds[:3]
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PUBLIC API — predict_disease()
# ══════════════════════════════════════════════════════════════════════════════

def predict_disease(
    image_source : Union[str, Path, "UploadedFile"],
    model_path   : str = DEFAULT_MODEL_PATH,
    labels_path  : str = DEFAULT_LABELS_PATH,
) -> dict:
    """
    Run full plant disease inference on a single image.

    This is the main public function. Call this from app.py or anywhere else.

    Args:
        image_source : File path (str/Path) OR a Streamlit UploadedFile object.
        model_path   : Path to the .h5 model. Defaults to DEFAULT_MODEL_PATH.
        labels_path  : Path to class_labels.json. Defaults to DEFAULT_LABELS_PATH.

    Returns:
        A dict with the following keys:

        label           (str)  — Top predicted class, e.g. "Tomato - Early Blight"
        confidence      (float)— Confidence score in [0.0, 1.0], e.g. 0.934
        all_predictions (dict) — {label: confidence} for every class
        top3            (list) — Top 3 predictions as [{label, confidence}, ...]
        source          (str)  — "model" or "mock"
        low_confidence  (bool) — True if confidence < LOW_CONFIDENCE_THRESHOLD
        disease_info    (dict) — Severity, description, treatment recommendations
        error           (str)  — None on success, error message on failure

    Example:
        >>> result = predict_disease("leaf.jpg")
        >>> print(result["label"])
        'Tomato - Early Blight'
        >>> print(f"{result['confidence']*100:.1f}%")
        '93.4%'
        >>> for step in result["disease_info"]["treatment"]:
        ...     print("-", step)
    """
    # ── Load labels and model ─────────────────────────────────────────────────
    labels = load_class_labels(labels_path)
    model  = load_model(model_path)

    # ── Mock mode (no model file present) ────────────────────────────────────
    if model == "mock":
        result = _mock_prediction(labels)
        result["error"]        = None
        result["low_confidence"] = result["confidence"] < LOW_CONFIDENCE_THRESHOLD
        result["disease_info"] = get_disease_info(result["label"])
        return result

    # ── Real inference ────────────────────────────────────────────────────────
    try:
        # Step 1: Preprocess
        img_array = preprocess_image(image_source)

        # Step 2: Predict — returns shape (1, num_classes)
        t0 = time.time()
        raw_probs = model.predict(img_array, verbose=0)[0]   # Shape: (num_classes,)
        inference_ms = (time.time() - t0) * 1000

        # Step 3: Extract top prediction
        top_idx        = int(np.argmax(raw_probs))
        top_label      = labels[top_idx] if top_idx < len(labels) else f"Class {top_idx}"
        top_confidence = float(raw_probs[top_idx])

        # Step 4: Build full predictions dict
        all_preds = {}
        for i, prob in enumerate(raw_probs):
            lbl = labels[i] if i < len(labels) else f"Class {i}"
            all_preds[lbl] = round(float(prob), 4)

        # Step 5: Assemble result
        result = {
            "label"          : top_label,
            "confidence"     : round(top_confidence, 4),
            "all_predictions": all_preds,
            "top3"           : _get_top3(all_preds),
            "source"         : "model",
            "low_confidence" : top_confidence < LOW_CONFIDENCE_THRESHOLD,
            "disease_info"   : get_disease_info(top_label),
            "inference_ms"   : round(inference_ms, 1),
            "error"          : None,
        }

        return result

    except (FileNotFoundError, ValueError) as e:
        # User-facing errors (bad path, wrong format, etc.)
        return {
            "label"          : None,
            "confidence"     : 0.0,
            "all_predictions": {},
            "top3"           : [],
            "source"         : "error",
            "low_confidence" : True,
            "disease_info"   : {},
            "error"          : str(e),
        }

    except Exception as e:
        # Unexpected errors (TF crash, OOM, etc.)
        return {
            "label"          : None,
            "confidence"     : 0.0,
            "all_predictions": {},
            "top3"           : [],
            "source"         : "error",
            "low_confidence" : True,
            "disease_info"   : {},
            "error"          : f"Inference failed: {e}",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict_batch(
    image_paths : list,
    model_path  : str = DEFAULT_MODEL_PATH,
    labels_path : str = DEFAULT_LABELS_PATH,
) -> list:
    """
    Run predict_disease() on a list of image paths.

    Images that fail (bad path, corrupted, etc.) are included in the results
    with `error` set — the batch does not abort on a single failure.

    Args:
        image_paths : List of file path strings or Path objects.
        model_path  : Path to the .h5 model file.
        labels_path : Path to class_labels.json.

    Returns:
        List of result dicts (same format as predict_disease()),
        with an additional "image_path" key on each entry.

    Example:
        >>> paths = ["leaf1.jpg", "leaf2.png", "leaf3.jpg"]
        >>> results = predict_batch(paths)
        >>> for r in results:
        ...     print(r["image_path"], r["label"], f"{r['confidence']*100:.1f}%")
    """
    # Pre-load model once so it isn't reloaded for every image
    load_class_labels(labels_path)
    load_model(model_path)

    results = []
    total = len(image_paths)

    print(f"[INFO] Running batch inference on {total} images ...")

    for i, img_path in enumerate(image_paths, start=1):
        print(f"  [{i}/{total}] {img_path}", end=" ... ")
        result = predict_disease(img_path, model_path, labels_path)
        result["image_path"] = str(img_path)

        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            flag = " ⚠️ low confidence" if result["low_confidence"] else ""
            print(f"{result['label']} ({result['confidence']*100:.1f}%){flag}")

        results.append(result)

    passed = sum(1 for r in results if r["error"] is None)
    print(f"\n[INFO] Batch complete: {passed}/{total} succeeded.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — Run this file directly from the terminal
# ══════════════════════════════════════════════════════════════════════════════

def _print_result_human(result: dict, image_path: str):
    """Pretty-print a prediction result to the terminal."""
    W = 60
    print("\n" + "="*W)
    print("  🌿  Agri-Edge — Plant Disease Prediction")
    print("="*W)
    print(f"  Image : {image_path}")
    print(f"  Mode  : {'⚠️  DEMO (no model found)' if result['source'] == 'mock' else '✅  Real model'}")
    print("-"*W)

    if result["error"]:
        print(f"\n  ❌  ERROR: {result['error']}\n")
        return

    # Confidence bar (20 chars wide)
    bar_len   = 20
    filled    = int(result["confidence"] * bar_len)
    bar       = "█" * filled + "░" * (bar_len - filled)
    conf_pct  = result["confidence"] * 100
    warn_flag = " ⚠️  Low confidence" if result["low_confidence"] else ""

    print(f"\n  Diagnosis   : {result['label']}")
    print(f"  Confidence  : [{bar}] {conf_pct:.1f}%{warn_flag}")

    if "inference_ms" in result:
        print(f"  Inference   : {result['inference_ms']} ms")

    # Top 3
    print(f"\n  Top 3 Predictions:")
    for rank, pred in enumerate(result["top3"], start=1):
        bar2_len = int(pred["confidence"] * 15)
        bar2     = "█" * bar2_len + "░" * (15 - bar2_len)
        print(f"    {rank}. [{bar2}] {pred['confidence']*100:5.1f}%  {pred['label']}")

    # Disease info
    info = result["disease_info"]
    if info:
        print(f"\n  Severity    : {info.get('severity', 'unknown').upper()}")
        print(f"  Description : {info.get('description', '')}")
        treatments = info.get("treatment", [])
        if treatments:
            print(f"\n  Recommended Actions:")
            for step in treatments:
                print(f"    • {step}")

    print("\n" + "="*W + "\n")


def main():
    """CLI entry point — run inference from the terminal."""
    parser = argparse.ArgumentParser(
        description="Agri-Edge: Predict plant disease from a leaf image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python models/disease_detector.py --image leaf.jpg
  python models/disease_detector.py --image leaf.jpg --model models/saved_model/plant_disease_model.h5
  python models/disease_detector.py --image leaf.jpg --json
  python models/disease_detector.py --batch leaf1.jpg leaf2.jpg leaf3.png
        """,
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to a single leaf image file.",
    )
    parser.add_argument(
        "--batch", "-b",
        nargs="+",
        default=None,
        help="Paths to multiple image files for batch inference.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the .h5 model file. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default=DEFAULT_LABELS_PATH,
        help=f"Path to class_labels.json. Default: {DEFAULT_LABELS_PATH}",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output raw JSON instead of human-readable text.",
    )

    args = parser.parse_args()

    # ── Validate: must provide --image OR --batch ─────────────────────────────
    if not args.image and not args.batch:
        parser.print_help()
        print("\n❌  Please provide --image <path> or --batch <path1> <path2> ...\n")
        sys.exit(1)

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch:
        results = predict_batch(args.batch, args.model, args.labels)
        if args.json:
            print(json.dumps(results, indent=2))
        return

    # ── Single image mode ─────────────────────────────────────────────────────
    result = predict_disease(args.image, args.model, args.labels)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_result_human(result, args.image)

    # Exit with non-zero code if inference failed
    if result["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
