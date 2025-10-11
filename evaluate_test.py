import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

# Keep imports consistent with training/model
from model_training import LearnableMel, BandSE, DotProductMHA  # for loading models with custom layers
from Wav import load_data_with_labels_from_wav

# ===================== Configurable Area =====================
BASE_DIR = r"your path"  # Modify to your test-set directory if needed

MODEL_DIR = ".h5"
MODEL_BASENAME = "your model"

# Inference settings
BATCH_SIZE = 16
THRESHOLD = 0.5  # binary classification threshold

# Whether to print per-file predicted probabilities and labels
PRINT_PER_FILE = False

# ===================== End Configurable Area =====================


def _safe_normalize(batch_wave: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-sample normalization consistent with training: x / max(|x|).
    Input shape (N, L); returns the same shape.
    """
    max_abs = np.max(np.abs(batch_wave), axis=1, keepdims=True)
    max_abs = np.where(max_abs < eps, eps, max_abs)
    return batch_wave / max_abs


def _load_models(model_dir: str, basename: str, num_expected: int = 1):
    """
    Load existing fold models.
    Returns: models (list[tf.keras.Model])
    """
    custom_objects = {
        'LearnableMel': LearnableMel,
        'BandSE': BandSE,
        'DotProductMHA': DotProductMHA
    }

    models = []
    for k in range(1, num_expected + 1):
        path = os.path.join(model_dir, f"{basename}{k}.h5")
        if os.path.exists(path):
            print(f"[INFO] Loading model: {path}")
            m = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)

            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
            models.append(m)
        else:
            print(f"[WARN] Not found: {path}")

    if len(models) == 0:
        raise FileNotFoundError(
            f"No fold models were found under {model_dir} (e.g., {basename}1.h5). "
            f"Please run the training script to save models first."
        )
    return models


def _ensemble_predict(models, X, batch_size=16):
    """
    Ensemble: average probabilities over multiple fold models.
    X shape: (N, L)
    Returns: prob (N, 1)
    """
    probs = []
    for i, m in enumerate(models, 1):
        p = m.predict(X, batch_size=batch_size, verbose=0)
        if p.ndim == 1:
            p = p[:, None]
        probs.append(p)
    return np.mean(probs, axis=0)


def main():
    # ========== 1) Load test set ==========
    if not os.path.exists(BASE_DIR):
        print(f"[ERROR] Test directory does not exist: {BASE_DIR}")
        return

    print(f"[INFO] Loading test data from: {BASE_DIR}")
    data_tf, labels_tf, filenames = load_data_with_labels_from_wav(BASE_DIR)
    X = data_tf.numpy()   # (N, L)
    y = labels_tf.numpy() # (N,)

    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)

    print(f"[INFO] Loaded: X.shape={X.shape}, y.shape={y.shape}, files={len(filenames)}")
    X = _safe_normalize(X)
    print(f"[INFO] After normalize: X.shape={X.shape}")

    # Quick stats
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    print(f"[INFO] #Samples={len(y)}  #Normal(0)={n_neg}  #Abnormal(1)={n_pos}")

    # ========== 2) Load fold models ==========
    models = _load_models(MODEL_DIR, MODEL_BASENAME, num_expected=5)

    # ========== 3) Predict ==========
    y_prob = _ensemble_predict(models, X, batch_size=BATCH_SIZE)  # (N, 1)

    # Optional: print per-file predictions
    if PRINT_PER_FILE:
        print("\n[Per-file predictions]")
        for fn, prob in zip(filenames, y_prob.ravel()):
            pred = int(prob >= THRESHOLD)
            print(f"{fn:40s}  prob={prob:.4f}  pred={pred}")


if __name__ == "__main__":
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print(f"[WARN] set_memory_growth failed: {e}")

    main()
