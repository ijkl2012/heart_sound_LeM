import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib

# Set global font to Times New Roman
matplotlib.rcParams["font.family"] = "Times New Roman"

from model_training import LearnableMel
# import Wav  your data path

FS = 2000  # sampling rate in Hz


# ----------------------------
# 1) Integrated Gradients (IG)
# ----------------------------
def integrated_gradients(model, x, baseline, target_class=None, steps=50):
    """
    Compute Integrated Gradients (IG).

    Args:
      model: a trained Keras model
      x: input sample to explain, shape (1, time)
      baseline: baseline input with the same shape as x (e.g., all zeros)
      target_class: for binary classification set to 0; if None, defaults to the predicted class
      steps: number of interpolation steps

    Returns:
      ig: tensor with the same shape as x, IG value per time sample
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

    if target_class is None:
        preds = model(x, training=False)
        target_class = 0

    accumulated_grads = tf.zeros_like(x)
    alphas = tf.linspace(0.0, 1.0, steps)

    for alpha in alphas:
        x_step = baseline + alpha * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(x_step)
            preds = model(x_step, training=False)
            target_value = preds[:, target_class]
        grads = tape.gradient(target_value, x_step)
        accumulated_grads += grads

    avg_grads = accumulated_grads / steps
    ig = (x - baseline) * avg_grads
    return ig


# ----------------------------
# 2) Discrete coloring helper
# ----------------------------
def discrete_colored_line(time_axis, y, importance_norm, threshold=0.5):
    """
    Color each line segment between adjacent samples based on normalized IG values:
      - if the mean of two adjacent importance_norm values >= threshold: color 'lightcoral'
      - otherwise: color 'blue'

    Args:
      time_axis: 1D array of time in seconds
      y: 1D array of signal amplitude
      importance_norm: 1D array, normalized absolute IG values in [0, 1]
      threshold: threshold to mark high-contribution regions

    Returns:
      A LineCollection that can be added to a matplotlib axis
    """
    points = np.array([time_axis, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = []
    for i in range(len(importance_norm) - 1):
        avg_val = (importance_norm[i] + importance_norm[i + 1]) / 2.0
        if avg_val >= threshold:
            colors.append("lightcoral")
        else:
            colors.append("blue")
    lc = LineCollection(segments, colors=colors, linewidth=2)
    return lc


# ----------------------------
# 3) Main: load model/data, compute IG, and plot
# ----------------------------
def main():
    # Model path
    model_path = "your model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please check the path!")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'LearnableMel': LearnableMel}
    )

    # Data path (modify to your data directory)
    base_dir = r"your path"
    data_tensor, labels_tensor, filenames = wav(base_dir)
    data = data_tensor.numpy()
    data = data / np.max(np.abs(data), axis=1, keepdims=True)

    num_display = min(10, data.shape[0])
    steps = 50
    threshold = 0.5  # IG normalization threshold

    for i in range(num_display):
        x = data[i:i + 1]
        baseline = np.zeros_like(x, dtype=np.float32)
        ig = integrated_gradients(model, x, baseline, target_class=None, steps=steps)
        ig_np = ig.numpy().squeeze()  # IG values, shape (time,)
        x_np = x.squeeze()            # raw signal, shape (time,)

        # Convert sample index to time (seconds)
        time_axis = np.arange(len(x_np)) / FS

        # Normalize absolute IG to [0, 1]
        importance = np.abs(ig_np)
        eps = 1e-8
        importance_norm = (importance - np.min(importance)) / (np.max(importance) - np.min(importance) + eps)

        lc = discrete_colored_line(time_axis, x_np, importance_norm, threshold=threshold)

        fig, ax = plt.subplots(figsize=(12, 4))
        # Plot background gray waveform
        ax.plot(time_axis, x_np, color='gray', linewidth=1, alpha=0.5)
        # Add line segments colored by IG
        ax.add_collection(lc)
        # Add colorbar (numeric ticks only)
        sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax)

        # X-axis ticks: [0, 0.5, 1.0, 1.5, 2.0]
        ax.set_xlim(0, 2.0)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])

        # Y-axis ticks: [-1.0, -0.5, 0, 0.5, 1.0]
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
