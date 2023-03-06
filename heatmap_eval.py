import torch
import numpy as np
# import os
import argparse
# import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import roc_auc_score, plot_roc_curve
import define_model
# from torch.optim import Adam
from plot_utils import compute_median_and_variance_roc_sic
from matplotlib import cm
from scipy.interpolate import interp1d
import data_process
import model_training
import model_evaluation

# choose the best 10 epochs
def choose_ten_best(x):
    index = np.argpartition(x,10)
    return np.ndarray.tolist(index[:10])

def evaluate_sic_max(device, dir=None, l1 = 64, l2 = 64, l3 =64, S_over_B="", data_reduc=1.0, signal_vs_bg=False, gauss_num=0, uniform_num=0, phi=False, dropout=False):
    print("From " + dir)
    # define parameters
    if phi:
        X_test = np.load("X_files/X_test_phi.npy")
        y_test = np.load("X_files/y_test_phi.npy")
    elif uniform_num>0:
        X_test = np.load("X_files/X_test_uni.npy")
        y_test = np.load("X_files/y_test_uni.npy")
        X_test = X_test[:, :4 + uniform_num]
    else:
        if signal_vs_bg:
            X_train = np.load("X_files/X_train_sb"+S_over_B+".npy")
            y_train = np.load("X_files/y_train_sb"+S_over_B+".npy")
            sig_num = model_evaluation.count_sig(y_train)
            print("Fully supervised")
        else:
            X_train = np.load("X_files/X_train"+S_over_B+".npy")
            y_train = np.load("X_files/y_train"+S_over_B+".npy")
            sig_num = model_evaluation.count_sig(y_train)
            print("Idealized AD")
        
        X_train, y_train = model_training.data_reduction(X_train, y_train, data_reduc=data_reduc, bg_vs_bg=False, signal_vs_bg=signal_vs_bg)
        X_test = np.load("X_files/X_test.npy")
        y_test = np.load("X_files/y_test.npy")
        X_test = data_process.data_normalize(X_test, X_train)
        X_test = X_test[:, :4 + gauss_num]
    
    X_db, y_db = model_evaluation.data_vs_bg_label(X_test, y_test, sig_num) # for test loss task (data vs bg)
    # Class weight
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_db[:, 0]), y=y_db[:, 0])
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    
    loss_fn = define_model.BCELoss_class_weighted(class_weights)

    # from np to torch
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    X_db = torch.from_numpy(X_db).float().to(device)
    y_db = torch.from_numpy(y_db).float().to(device)

    tprs_list = []
    fprs_list = []
    auc_list = []
    mean_loss = []
    collect_val = [] # for median on the legends

    input_number = len(X_test[0])
    num_model = 10

    for model_num in range(num_model):
        # defining new NN every change of model
        if dropout == True:
            model = define_model.NeuralNetworkDropout(input_number).to(device)
        else:
            model = define_model.NeuralNetwork(input_number, l1, l2, l3).to(device)

        tpr = []
        fpr =[]
        collect_pred = []
        collect_loss = []

        val_losses = np.load("saved_model/"+dir+"val_losses_model_"+str(model_num+1)+".npy")

        model_number = choose_ten_best(val_losses)

        # collect val values for median on legends
        val_list = [] # for mean of val losses to pass to collect_val
        for val in model_number:
            val_list.append(val_losses[val])

        collect_val.append(np.mean(val_list))

        for i in model_number:

            model.load_state_dict(torch.load("saved_model/"+dir+"saved_training_model"+str(model_num+1)+'_epoch'+str(i+1)+".pt"))

            # Evaluation
            model.eval()
            test_pred = []
            test_loss = 0
            with torch.no_grad():
                # signal vs bg
                test_pred = model(X_test) # model() return probability [0,1]

                # data vs bg
                db_pred = model(X_db)
                test_loss = loss_fn(db_pred, y_db)
            collect_loss.append(test_loss.cpu().item())
            collect_pred.append(test_pred.cpu().numpy())
        
        mean_loss.append(np.mean(collect_loss, axis = 0))
        collect_pred = np.mean(collect_pred, axis = 0)

        fpr, tpr, _ = roc_curve(y_test.cpu(), collect_pred)
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)
        tprs_list.append(tpr)
        fprs_list.append(fpr)

    _, _, sic_median, _, sic_std = compute_median_and_variance_roc_sic(
        tprs_list, fprs_list, resolution=1000, fpr_cutoff=0.000052777777778)
    sic_max = np.amax(sic_median)
    sic_lower = sic_std[0][int(np.argmax(sic_median))]
    sic_upper = sic_std[1][int(np.argmax(sic_median))]
    # make it strictly positive (the lowest now is -40)
    test_median = (np.median(mean_loss)-0.69)*1e3
    test_spread = np.nanquantile(mean_loss, 0.84) - np.nanquantile(mean_loss, 0.16)
    test_spread = (test_spread/2)*1e3
    return sic_max, sic_lower, sic_upper, test_median, test_spread