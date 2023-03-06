from xml.dom import minicompat
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, plot_roc_curve
from scipy.interpolate import interp1d

def compute_median_and_variance_roc_sic(tprs_list, fprs_list, resolution=1000, low_quatile=0.16, high_quatile=0.84, extended_range=False, fpr_cutoff=0):
    # interpolation
    max_min_tpr = 0.
    min_max_tpr = 1.
    if not extended_range:
        for tpr in tprs_list:
            if min(tpr) > max_min_tpr:
                max_min_tpr = min(tpr)
            if max(tpr) < min_max_tpr:
                min_max_tpr = max(tpr)
    tpr_manual = np.linspace(max_min_tpr, min_max_tpr, resolution)

    roc_interpol = []
    sic_interpol = []
    for tpr, fpr in zip(tprs_list, fprs_list):
        fpr_mask = fpr>fpr_cutoff
        tpr_vals = tpr[fpr_mask]
        fpr_vals = fpr[fpr_mask]
        if len(tpr_vals)>2:
            roc_function = interp1d(tpr_vals, 1/fpr_vals, bounds_error=False)
            sic_function = interp1d(tpr_vals, tpr_vals/(fpr_vals**(0.5)), bounds_error=False)
            roc_interpol.append(roc_function(tpr_manual))
            sic_interpol.append(sic_function(tpr_manual))
        else:
            roc_function = 1/tpr_manual
            sic_function = tpr_manual/np.sqrt(tpr_manual)
            roc_interpol.append(roc_function)
            sic_interpol.append(sic_function)
    
    # ensure at least 5 lines for a point to be considered
    roc_matrix = np.stack(roc_interpol)
    sic_matrix = np.stack(sic_interpol)
    roc_line_count = np.count_nonzero(~np.isnan(roc_matrix), axis = 0)
    sic_line_count = np.count_nonzero(~np.isnan(sic_matrix), axis = 0)
    total_count = np.min(np.stack((roc_line_count, sic_line_count)), axis = 0)
    roc_matrix = roc_matrix[:, total_count>=5]
    sic_matrix = sic_matrix[:, total_count>=5]
    tpr_manual = tpr_manual[total_count>=5]

    # calculating median and quantiles
    roc_median = np.nanmedian(roc_matrix, axis = 0)
    sic_median = np.nanmedian(sic_matrix, axis = 0)
    roc_std = (np.nanquantile(roc_matrix, low_quatile, axis = 0), np.nanquantile(roc_matrix, high_quatile, axis = 0))
    sic_std = (np.nanquantile(sic_matrix, low_quatile, axis = 0), np.nanquantile(sic_matrix, high_quatile, axis = 0))
    
    return tpr_manual, roc_median, sic_median, roc_std, sic_std 
