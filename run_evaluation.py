import torch
import model_evaluation
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm

parser = argparse.ArgumentParser(
    description=("Evaluating with or without epoch ensembling."), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mjj", action="store_true", default=False,
                    help="use mjj")
parser.add_argument("--signal_vs_bg", action="store_true", default=False,
                    help="train over signal vs background")
parser.add_argument("--epochs", type=int, default=100,
                    help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128,
                    help="number of batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3,
                    help="configuring learning rate")
args = parser.parse_args()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
learning_rate = args.learning_rate


# methods = [("baseline", 0), ("1G", 1), ("2G", 2), ("3G", 3), ("5G", 5), ("7G", 7), ("10G", 10)]

methods = [("2Gnodes4", 4),("2Gnodes8", 8), ("2Gnodes16", 16), ("2Gnodes32", 32), ("2G", 64), ("2Gnodes128", 128), ("2Gnodes256", 256), ("2Gnodes512", 512), ("2Gnodes1024", 1024)]
# methods = ["8", "16", "32", "64", "128", "256", "512", "1028", "0"]
# methods = [("2Glr0.0001", "1e-4"), ("2G", "1e-3"), ("2Glr0.002", "2e-3"), ("2Glr0.005", "5e-3"), ("2Glr0.05", "5e-2")]

# normal plot
# colors = cm.viridis(np.linspace(0., 0.95, len(methods)))
colors = cm.viridis(np.linspace(0., 0.95, 11))

plt.figure(figsize=(8, 6))

# model_evaluation.evaluate_tpr_fpr(device,"baseline/", gauss_num = 0, signal_vs_bg=False, label = "baseline", color = colors[0])
# model_evaluation.evaluate_tpr_fpr(device,"2G/", gauss_num = 2, signal_vs_bg=False, label = "2G", color = colors[3])
# model_evaluation.evaluate_tpr_fpr(device,"0G_dropout/", gauss_num = 0, dropout=True, signal_vs_bg=False, label = "0G dropout", color = colors[6])
# model_evaluation.evaluate_tpr_fpr(device,"2G_dropout/", gauss_num = 2, dropout=True, signal_vs_bg=False, label = "2G dropout", color = colors[9], with_random=True)

# model_evaluation.evaluate_tpr_fpr(device,"0G_sgd/", gauss_num = 0, signal_vs_bg=False, label = "SGD", color = colors[2])
# model_evaluation.evaluate_tpr_fpr(device,"0G_sgd_momentum05/", gauss_num = 0, signal_vs_bg=False, label = "SGD momentum = 0.5", color = colors[4])
# model_evaluation.evaluate_tpr_fpr(device,"0G_sgd_momentum/", gauss_num = 0, signal_vs_bg=False, label = "SGD momentum = 0.9", color = colors[6], with_random=True)
# model_evaluation.evaluate_tpr_fpr(device,"1G/", gauss_num = 1, signal_vs_bg=False, label = "1G", color = colors[2])
# model_evaluation.evaluate_tpr_fpr(device,"2G/", gauss_num = 2, signal_vs_bg=False, label = "2G", color = colors[4])
# model_evaluation.evaluate_tpr_fpr(device,"0G_tune/", l1=32, l2=64, l3=16, gauss_num = 0, signal_vs_bg=False, label = "0G tune", color = colors[6])
# model_evaluation.evaluate_tpr_fpr(device,"1G_tune/", l1=64, l2=32, l3=64, gauss_num = 1, signal_vs_bg=False, label = "1G tune", color = colors[8])
# model_evaluation.evaluate_tpr_fpr(device,"2G_tune/", l1=4, l2=153, l3=150, gauss_num = 2, signal_vs_bg=False, label = "2G tune", color = colors[10], with_random=True)

# model_evaluation.evaluate_tpr_fpr(device,"0Gallhyperparam/", l1=512, l2=512, l3=512, gauss_num = 0, signal_vs_bg=False, label = "0G 512 nodes, batch 16, 2e-3 lr", color = colors[0])
# model_evaluation.evaluate_tpr_fpr(device,"0G_batch16_lr2e-3/", gauss_num = 0, signal_vs_bg=False, label = "0G batch 16, 2e-3 lr", color = colors[3])
# model_evaluation.evaluate_tpr_fpr(device,"0G_batch16_nodes512/", l1=512, l2=512, l3=512, gauss_num = 0, signal_vs_bg=False, label = "0G 512 nodes, batch 16", color = colors[6])
# model_evaluation.evaluate_tpr_fpr(device,"0Gnodes512lr2e-3/", l1=512, l2=512, l3=512, gauss_num = 0, signal_vs_bg=False, label = "0G 512 nodes, 2e-3 lr", color = colors[8], with_random=True)
# model_evaluation.evaluate_tpr_fpr(device,"2Gnodes512lr2e-3/", l1=512, l2=512, l3=512, gauss_num = 2, signal_vs_bg=False, label = "2G 512 nodes 2e-3 lr", color = colors[9], with_random=True)

# # model_evaluation.evaluate_tpr_fpr(device,"0G_hingeloss/", gauss_num = 0, hingeloss=True, signal_vs_bg=False, label = "Hinge loss", color = colors[5], with_random=True)
# model_evaluation.evaluate_tpr_fpr(device,"1G/", gauss_num = 1, signal_vs_bg=False, label = "1G", color = colors[3])
# model_evaluation.evaluate_tpr_fpr(device,"phij1/", phi=True, signal_vs_bg=False, label = "$\\phi_{J_1}$", color = colors[6])
# model_evaluation.evaluate_tpr_fpr(device,"1U/", uniform_num=1, signal_vs_bg=False, label = "1U", with_random=True, color = colors[9])

# model_evaluation.evaluate_tpr_fpr(device,"EPiC/", gauss_num = 0, use_EPiC=True, signal_vs_bg=False, label = "EPiC", with_random=True, color = "green")
model_evaluation.evaluate_tpr_fpr_single_epoch(device,"EPiC_signal_vs_bg/", epoch = -1, use_EPiC=True, signal_vs_bg=True, label = "EPiC min epoch", color = "black")
model_evaluation.evaluate_tpr_fpr_single_epoch(device,"EPiC_signal_vs_bg/", epoch = 40, use_EPiC=True, signal_vs_bg=True, label = "EPiC 40th epoch", color = "maroon")
model_evaluation.evaluate_tpr_fpr_single_epoch(device,"EPiC_signal_vs_bg/", epoch = 149, use_EPiC=True, signal_vs_bg=True, label = "EPiC last epoch", color = "green", with_random=True)

# count = 0
# for method, node in methods:
#     # node = 64
#     dir = method + "/"
#     gauss = 2
#     signal_vs_bg = False
#     with_random = False
#     l1 = node
#     l2 = node
#     l3 = node
#     if count == len(methods)-1:
#         with_random = True
#     model_evaluation.evaluate_tpr_fpr(device, dir, l1=l1, l2=l2,l3=l3, gauss_num = gauss, label = str(node) + " nodes", signal_vs_bg=signal_vs_bg, color = colors[count], batch_size=0, with_random=with_random)
#     count += 1
plt.subplots_adjust(right=2.0)
plt.savefig("plots/results/test.pdf", bbox_inches="tight")
plt.close()

# different S/B
# methods = [("", "100% data", 1), ("080", "80% data", 1),("050", "50% data", 1), ("030", "30% data", 1), ("020", "20% data", 1), ("015", "15% data", 1), ("010", "10% data", 1)]
# methods = [("", "100% data", 2)]
# sbs = [("006", "0.6% S/B "), ("007", "0.7% S/B "), ("009", "0.9% S/B "), ("015", "1.5% S/B "), ("020", "2.0% S/B "), ("030", "3.0% S/B "), ("040", "4.0% S/B "), ("050", "5.0% S/B "), ("100", "10.0% S/B "), ("150", "15.0% S/B "), ("250", "25.0% S/B ")]
# colors = cm.viridis(np.linspace(0., 0.95, len(methods)))

# for (sb, lb) in sbs:
#     plt.figure(figsize=(8, 6))
#     count = 0
#     for (method, label, gauss) in methods:
#         dir = "2Gcombo" + sb + method + "/"
#         label = lb + label
#         node = 512
#         if count == len(sb)-1:
#             model_evaluation.evaluate_tpr_fpr(device, dir, l1=node, l2=node, l3=node, S_over_B=sb, gauss_num = gauss, label = label, color = colors[count], with_random=True)
#         else:
#             model_evaluation.evaluate_tpr_fpr(device, dir, l1=node, l2=node, l3=node, S_over_B=sb, gauss_num = gauss, label = label, color = colors[count])
#         count += 1

# plt.subplots_adjust(right=2.0)
# plt.savefig("all_plots/different_S_B"+sb+"_2G.pdf", bbox_inches="tight")
# plt.close()

######### batch study
# model_evaluation.evaluate_batchid(device, "two_gauss/", name="loss_batchid_short_2G", gauss_num=2, color="black")