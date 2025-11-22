#src/cavl_doc/evaluation/metrics.py

from sklearn.metrics import roc_curve
import numpy as np

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    thr = thresholds[idx]
    return eer, thr