import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
# import os
import argparse
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import roc_auc_score, plot_roc_curve
import define_model
# from torch.optim import Adam
from plot_utils import compute_median_and_variance_roc_sic
from plot_utils import roc_curve as roc
from matplotlib import cm
from scipy.interpolate import interp1d
import data_process
import random
import model_training

# choose the best 10 epochs
def choose_ten_best(x):
    index = np.argpartition(x,10)
    return np.ndarray.tolist(index[:10])

def count_sig(y):
    sig_num = 0
    for i in y[:, 0]:
        if i == 1:
            sig_num += 1
    return sig_num

# there's no torch.random.choice 
def torch_random_choice(x, amount, device):
    # x is torch.tensor
    new_x = []
    index = torch.randint(0, len(x), (amount,))
    for i in index:
        new_x.append(x[i])
    return torch.stack(new_x).to(device)

def data_vs_bg_label(x, y, sig_num=369):
    x_sig = []
    x_bg = []
    count = 0
    for i in y:
        if i == 1:
            x_sig.append(x[count])
            count += 1
        else:
            x_bg.append(x[count])
            count += 1 
    np.stack(x_sig)
    np.stack(x_bg)
    x_data = np.concatenate((x_sig[:sig_num], x_bg[:(60000 - sig_num)]), axis = 0)
    x_bg_bg = x_bg[60001 - sig_num: 136429+60001 - sig_num]
    y_data = np.ones((len(x_data), 1))
    y_bg = np.zeros((len(x_bg_bg), 1))
    new_x = np.concatenate((x_data, x_bg_bg), axis=0)
    new_y = np.concatenate((y_data, y_bg), axis=0)
    return new_x, new_y

def evaluate_tpr_fpr(device, dir=None, l1=64, l2=64, l3=64, gauss_num=0, S_over_B="", data_reduc=1.0, signal_vs_bg=False, hingeloss=False, uniform_num=0, phi=False, duplicate_feature=0, dropout=False, morelayers=False, label="default", linestyle="-", color="black", with_random=False, fpr_cutoff=0.000052777777778):
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
            sig_num = count_sig(y_train)
            print("Fully supervised")
        else:
            X_train = np.load("X_files/X_train"+S_over_B+".npy")
            y_train = np.load("X_files/y_train"+S_over_B+".npy")
            sig_num = count_sig(y_train)
            print("Idealized AD")
        
        X_train, y_train = model_training.data_reduction(X_train, y_train, data_reduc=data_reduc, bg_vs_bg=False, signal_vs_bg=signal_vs_bg)
        pre_X_test = np.load("X_files/X_test.npy")
        y_test = np.load("X_files/y_test.npy")
        X_test = data_process.data_normalize(pre_X_test, X_train)
        X_test = X_test[:, :4 + gauss_num]
    
    if duplicate_feature>0:
        X_test = data_process.duplicate_feature(x=X_test, input_num=duplicate_feature)
    if hingeloss:
        y_test[y_test==0] = -1

    X_db, y_db = data_vs_bg_label(X_test, y_test, sig_num) # for test loss task (data vs bg)
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
    val_median = 0.
    test_median = 0.
    auc_median = 0.
    val_spread = 0.
    test_spread = 0.

    input_number = len(X_test[0])

    for model_num in range(10):
        print(f"Model {model_num+1}")
        # defining new NN every change of model
        if dropout:
            model = define_model.NeuralNetworkDropout(input_number).to(device)
        elif morelayers:
            model = define_model.NeuralNetworkMoreLayers(input_number).to(device)
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
            test_loss, test_correct = 0, 0
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

    tpr_manual, roc_median, sic_median, roc_std, sic_std = compute_median_and_variance_roc_sic(
        tprs_list, fprs_list, resolution=1000, fpr_cutoff=fpr_cutoff)  # 0.0000326797385620915032679738562091503267973856209150326797385620
    auc_median = np.median(auc_list)
    val_median = (np.median(collect_val)-0.69)*1e3
    test_median = (np.median(mean_loss)-0.69)*1e3
    val_spread = np.nanquantile(collect_val, 0.84) - np.nanquantile(collect_val, 0.16)
    val_spread = (val_spread/2)*1e3
    test_spread = np.nanquantile(mean_loss, 0.84) - np.nanquantile(mean_loss, 0.16)
    test_spread = (test_spread/2)*1e3

    # plot
    # random
    if with_random:
        tpr_random = np.arange(0, 1, 1/100000)
        roc_random = 1/tpr_random
        sic_random = tpr_random/np.sqrt(tpr_random)

    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(tpr_manual, roc_median, color = color, linestyle = linestyle, label = label + ', AUC: %0.3f' % auc_median)
    plt.fill_between(tpr_manual, roc_std[0], roc_std[1], color=color, alpha=0.3)
    if with_random:
        plt.plot(tpr_random, roc_random, 'k:', label="Random")

    plt.legend(loc = 'upper right')
    plt.yscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('1/False Positive Rate', fontsize=12)
    plt.xlabel('True Positive Rate', fontsize=12)

    # SIC
    plt.subplot(1, 2, 2)
    plt.plot(tpr_manual, sic_median, color = color, linestyle = linestyle, label = r"" + label + ', val loss: %0.2f' % val_median + '$\pm$%0.2f' % val_spread + ', test loss: %0.2f' % test_median + '$\pm$%0.2f' % test_spread)
    plt.fill_between(tpr_manual, sic_std[0], sic_std[1], color=color, alpha=0.3)
    if with_random:
        plt.plot(tpr_random, sic_random, 'k:', label="Random")

    plt.legend(loc = 'upper right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Significance Improvement', fontsize=12)
    plt.xlabel('True Positive Rate', fontsize=12)

def evaluate_batchid(device, dir=None, name="loss_batchid_short", l1=64, l2=64, l3=64, gauss_num=0, S_over_B="", signal_vs_bg=False, uniform_num=0, phi=False, duplicate_feature=0, dropout=False, morelayers=False, label="default", linestyle="-", color="black", with_random=False, fpr_cutoff=0.000052777777778):
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
            print("Fully supervised")
        else:
            X_train = np.load("X_files/X_train"+S_over_B+".npy")
            print("Idealized AD")
        pre_X_test = np.load("X_files/X_test.npy")
        y_test = np.load("X_files/y_test.npy")
        X_test = data_process.data_normalize(pre_X_test, X_train)
        X_test = X_test[:, :4 + gauss_num]
    
    if duplicate_feature>0:
        X_test = data_process.duplicate_feature(x=X_test, input_num=duplicate_feature)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    input_number = len(X_test[0])
    loss_fn = torch.nn.BCELoss()

    test_data = data_process.MakeASet(X_test, y_test)
    batch_size = 18
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # defining new NN every change of model
    if dropout:
        model = define_model.NeuralNetworkDropout(input_number).to(device)
    elif morelayers:
        model = define_model.NeuralNetworkMoreLayers(input_number).to(device)
    else:
        model = define_model.NeuralNetwork(input_number, l1, l2, l3).to(device)

    tpr = []
    fpr =[]
    collect_pred = []

    model_number = random.randint(1,10)

    val_losses = np.load("saved_model/"+dir+"val_losses_model_"+str(model_number)+".npy")

    epoch_number = np.argmin(val_losses)
    model.load_state_dict(torch.load("saved_model/"+dir+"saved_training_model"+str(model_number)+'_epoch'+str(epoch_number+1)+".pt"))

    # Evaluation
    model.eval()
    test_pred = []
    test_loss, test_correct = 0, 0
    loss_vec = [] # collect loss values
    count_one_vec = [] # collect numbers of y=1 in each batch
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            count_one = 0
            for k in y:
                if k==1:
                    count_one += 1
            test_pred = model(X)
            test_loss = loss_fn(test_pred, y)
            loss_vec.append(test_loss.cpu().item())
            count_one_vec.append(count_one)
    
    print("min loss: " + str(np.amin(loss_vec)) + ", # of signals: " + str(count_one_vec[np.argmin(loss_vec)]))
    print("max loss: " + str(np.amax(loss_vec)) + ", # of signals: " + str(count_one_vec[np.argmax(loss_vec)]))
    
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(range(math.ceil(len(y_test)/batch_size)), loss_vec)
    axs[0].set_ylabel("loss")
    axs[0].set_xlim(0,100)
    axs[1].plot(range(math.ceil(len(y_test)/batch_size)), count_one_vec)
    axs[1].set_ylabel("# of signals")
    axs[1].set_xlabel("Batch ID")
    axs[1].set_xlim(0,100)
    plt.savefig("all_plots/"+name+".pdf", bbox_inches="tight")
    plt.close()