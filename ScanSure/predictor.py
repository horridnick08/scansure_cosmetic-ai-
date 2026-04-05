"""
predictor.py — Model-based Real/Fake prediction for ScanSure
=============================================================
Loads scansure_classifier.keras (trained Keras model) and converts
OCR-extracted label data into those features.
"""

import os
import numpy as np

# ── Load model once at import time ──────────────────────────────────────────
_MODEL_PATH = r"C:\Users\rohit\Downloads\ss zip\ScanSure\New folder\scansure_classifier.keras"
_model = None

def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Keras model not found at {_MODEL_PATH}. "
                "Please ensure scansure_classifier.keras is in the New folder."
            )
        try:
            import tensorflow as tf
            # Load the Keras model
            _model = tf.keras.models.load_model(_MODEL_PATH)
        except ImportError:
            raise ImportError("tensorflow is not installed. Please run: pip install tensorflow")
        except Exception as e:
            raise RuntimeError(f"Error loading Keras model: {str(e)}")
    return _model


def extract_features(ocr_data: dict) -> np.ndarray:
    """Convert OCR-extracted dict → 1×6 feature array."""
    has_brand    = 1.0 if ocr_data.get("brand")    else 0.0
    has_product  = 1.0 if ocr_data.get("product")  else 0.0
    has_barcode  = 1.0 if ocr_data.get("barcode")  else 0.0
    has_batch    = 1.0 if ocr_data.get("batch")    else 0.0

    ingredients = ocr_data.get("ingredients") or []
    ingredient_count_norm = min(len(ingredients), 10) / 10.0

    raw_text = ocr_data.get("raw_text") or ""
    text_length_norm = min(len(raw_text), 500) / 500.0

    # Ensure shape is (1, 6) for Keras
    features = np.array([[
        has_brand,
        has_product,
        has_barcode,
        has_batch,
        ingredient_count_norm,
        text_length_norm,
    ]], dtype=np.float32)
    return features


def predict(ocr_data: dict) -> dict:
    """
    Runs the Keras model on OCR data.
    If the model expects a different shape (e.g. image input), logs error and falls back.
    """
    try:
        model = _load_model()
        features = extract_features(ocr_data)

        # ── Step 1: Run prediction ──────────────────────────────────────────
        # model.predict returns a 2D array: (1, n_classes)
        preds = model.predict(features, verbose=0) 
        
        # ── Step 2: Interpret results ───────────────────────────────────────
        if preds.shape[1] == 1:
            # Binary sigmoid output [prob_of_real]
            confidence = float(preds[0][0])
            label_int = 1 if confidence >= 0.5 else 0
            if label_int == 0: 
                confidence = 1.0 - confidence # Return probability of 'Fake' if that's the label
        else:
            # Softmax output [prob_fake, prob_real]
            label_int = int(np.argmax(preds[0]))
            confidence = float(preds[0][label_int])

        return {
            "label":      "Real" if label_int == 1 else "Fake",
            "confidence": round(confidence, 4),
            "source":     "keras_model"
        }

    except Exception as e:
        # Graceful fallback: rule-based heuristic
        feats = extract_features(ocr_data)
        score = float(np.mean(feats))          # avg of 6 binary/norm features
        label = "Real" if score >= 0.4 else "Fake"
        
        error_msg = str(e)
        # Simplify common Keras value errors for readability
        if "input" in error_msg.lower() and "shape" in error_msg.lower():
            error_msg = "Model input shape mismatch. Is this an image model?"

        return {
            "label":      label,
            "confidence": round(score, 4),
            "source":     f"heuristic (Error: {error_msg})" 
        }
