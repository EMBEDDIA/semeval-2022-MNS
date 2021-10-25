from scipy.stats import pearsonr
import numpy as np

def evaluate_scores(pred_scores, true_scores):
    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    pearsonr_corr = pearsonr(pred_scores, true_scores)
    print("Pearson-r:", pearsonr_corr[0])
    print("p-value:", pearsonr_corr[1])
    return pearsonr_corr
