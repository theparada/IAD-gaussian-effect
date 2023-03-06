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
import data_process
# from torch.optim import Adam
from plot_utils import compute_median_and_variance_roc_sic
from matplotlib import cm

parser = argparse.ArgumentParser(
    description=("Evaluating with or without epoch ensembling."), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mjj", action="store_true", default=False,
                    help="use mjj")
parser.add_argument("--signal_vs_bg", action="store_true", default=False,
                    help="train over signal vs background")
parser.add_argument("--test_data_bg", action="store_true", default=False,
                    help="test over data vs background")
args = parser.parse_args()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# choose the best 10 epochs
def choose_ten_best(x):
    index = np.argpartition(x,10)
    return np.ndarray.tolist(index[:10])

def evaluate_tpr_fpr(dir=None, l1 = 64, l2 = 64, l3 =64, seed = 0, signal_vs_bg=False, gauss_num=0, duplicate_feature=0, test_data_bg=args.test_data_bg, dropout=False, label="default", linestyle="-", color="black"):
    # define parameters
    if signal_vs_bg:
        X_test = np.load("X_files/X_test_sb.npy")
        y_test = np.load("X_files/y_test_sb.npy")
    elif test_data_bg:
        # evaluate on validation set to see randomness
        X_test = np.load("X_files/X_val.npy")
        y_test = np.load("X_files/y_val.npy")
    else:
        if seed != 0:
            X_test = np.load('X_files/initialized_seed/X_test_'+str(seed)+'.npy')
            y_test = np.load('X_files/initialized_seed/y_test_'+str(seed)+'.npy')
            print("use seed")
        else:
            X_test = np.load("X_files/X_test.npy")
            y_test = np.load("X_files/y_test.npy")

    X_test = X_test[:, :4 + gauss_num]
    X_test = data_process.duplicate_feature(x=X_test, input_num=duplicate_feature)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # In case of evaluating data vs bg, it is faster to define X and y here
    X, y = X_test, y_test
    # X, y = X_val, y_val

    print("From " + dir)

    collect_val = [] # for median on the legends
    output = []
    sig = []
    bg = []

    input_number = len(X_test[0])
    print("Running " + str(input_number - 4) + " gauss(es) models.")
    loss_fn = torch.nn.BCELoss()
    num_model = 10

    for model_num in range(num_model):

        print(f"Model {model_num+1}\n-------------------------------")
        # defining new NN every change of model
        if dropout == True:
            model = define_model.NeuralNetworkDropout(input_number).to(device)
        else:
            model = define_model.NeuralNetwork(input_number, l1, l2, l3).to(device)

        tpr = []
        fpr =[]
        collect_pred = []

        val_losses = np.load("saved_model/"+dir+"val_losses_model_"+str(model_num+1)+".npy")

        model_number = choose_ten_best(val_losses)

        # collect val values for median on legends
        val_list = [] # for mean of val losses to pass to collect_val
        for val in model_number:
            val_list.append(val_losses[val])

        collect_val.append(np.mean(val_list))

        for i in model_number:
            print(f"Evaluating Epoch {i+1}\n-------------------------------")

            model.load_state_dict(torch.load("saved_model/"+dir+"saved_training_model"+str(model_num+1)+'_epoch'+str(i+1)+".pt"))

            # Evaluation
            model.eval()
            test_pred = []
            test_loss, test_correct = 0, 0
            with torch.no_grad():
                test_pred = model(X)
                test_loss = loss_fn(test_pred, y)
                test_correct = (test_pred.round() == y).sum().item()
                test_correct /= len(X)

            collect_pred.append(test_pred.cpu().numpy())
            print(f"Test Result: \n Accuracy: {(100*test_correct):>0.1f}%, loss: {test_loss:>8f} \n")
            print("Evaluation is Done! \n")

        collect_pred = np.mean(collect_pred, axis = 0)
        output.append(collect_pred)
    output = np.median(output, axis = 0)
    output = np.concatenate((y_test.cpu().numpy(), output), axis = 1)
    for (y, out) in output:
        if y == 1.:
            sig.append(out)
        else:
            bg.append(out)
    plt.subplot(1, 2, 1)
    plt.hist(sig, bins = 80, histtype='step', range=(0.46, 0.94), color=color, label=label + " sig", linestyle="solid")
    plt.legend(loc = 'upper right')
    plt.xlabel("Output")
    plt.yscale('log')
    plt.ylabel("Events")

    plt.subplot(1, 2, 2)
    plt.hist(bg, bins = 80, histtype='step', range=(0.46, 0.94), color=color, label=label + " bg", linestyle="solid")
    plt.legend(loc = 'upper right')
    plt.xlabel("Output")
    plt.yscale('log')
    plt.ylabel("Events")

methods = [("default", "default", 0), ("one_gauss", "1G", 1), ("two_gauss", "2G", 2), ("three_gauss", "3G", 3), ("seven_gauss","7G", 7), ("ten_gauss", "10G", 10)]
# methods = [("default_tune_l2_init", "tuned default L2", 0, 32, 65, 79), ("1G_tune_l2_init","tuned 1G L2", 1, 159, 16, 217), ("2G_tune_l2_init", "tuned 2G L2", 2, 29, 207, 159), ("3G_tune_l2_init", "tuned 3G L2", 3, 80, 78, 221), ("7G_tune_l2_init","tuned 7G L2", 7, 22, 201, 224), ("10G_tune_l2_init", "tuned 10G L2", 10, 4, 221, 202)]


colors = cm.viridis(np.linspace(0., 0.95, len(methods)))
count = 0

plt.figure(figsize=(8, 6))
for (method, label, gauss) in methods:
    dir = method + "/"
    # evaluate_tpr_fpr(dir, l1=l1, l2=l2, l3=l3, gauss_num = gauss, label = label, color=colors[count])
    evaluate_tpr_fpr(dir, gauss_num = gauss, label = label, color=colors[count])
    count += 1

plt.subplots_adjust(right=2.0)
plt.savefig("all_plots/eval_hist.png", bbox_inches="tight")
plt.close()