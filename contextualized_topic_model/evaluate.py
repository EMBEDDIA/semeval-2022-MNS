from scipy.stats import pearsonr, entropy
import numpy as np


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def compute_kld(p, q):
    return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))


def compute_kld2(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def evaluate_scores(pred_scores, true_scores):
    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    pearsonr_corr = pearsonr(pred_scores, true_scores)
    print("Pearson-r:", pearsonr_corr[0])
    print("p-value:", pearsonr_corr[1])
    return pearsonr_corr
