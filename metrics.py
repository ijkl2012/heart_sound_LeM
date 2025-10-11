import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, f1_score


def _evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5):
    """
     Acc/Recall/Specificity/F1
    y_true: (N,), 0/1
    y_prob: (N,1) or (N,)
    """
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob = y_prob.ravel()
    y_pred = (y_prob >= thresh).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    recall = recall_score(y_true, y_pred)  # Sensitivity
    specificity = tn / max(1, (tn + fp))
    f1 = f1_score(y_true, y_pred)

    return {
        "acc": acc,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "cm": (tn, fp, fn, tp),
        "y_pred": y_pred
    }