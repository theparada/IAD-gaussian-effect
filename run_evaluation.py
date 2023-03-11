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


methods = [("baseline", "Baseline", 0)]#, ("1G", "1G", 1), ("2G", "2G", 2), ("3G", "3G", 3), ("5G", "5G", 5), ("7G", "7G", 7), ("10G", "10G", 10)]
# methods = [("default_16nodes", "16 nodes", 16, 0), ("default_32nodes", "32 nodes", 32, 0), ("default", "64 nodes", 64, 0), ("default_128nodes", "128 nodes", 128, 0), ("default_256nodes", "256 nodes", 256, 0), ("default_512nodes", "512 nodes", 512, 0), ("default_1024nodes", "1024 nodes", 1024, 0)]
# methods = [("1G_16nodes", "16 nodes", 16, 1), ("1G_32nodes", "32 nodes", 32, 1), ("one_gauss","64 nodes", 64, 1), ("1G_128nodes", "128 nodes", 128, 1), ("1G_256nodes", "256 nodes", 256, 1)]
# methods = [("2G_16nodes", "16 nodes", 16, 2), ("2G_32nodes", "32 nodes", 32, 2), ("two_gauss", "64 nodes", 64, 2),
#            ("2G_128nodes", "128 nodes", 128, 2), ("2G_256nodes", "256 nodes", 256, 2), ("2G_512nodes", "512 nodes", 512, 2), ("2G_1024nodes", "1024 nodes", 1024, 2)]
# methods = [("default", "Vanilla", False, 0), ("default_dropout","Vanilla dropout", True, 0), ("two_gauss", "2G", False, 2), ("two_gauss_dropout", "2G dropout", True, 2)]
# methods = [("_2e-4", "2e-4"), ("", "1e-3"), ("_2e-3", "2e-3"), ("_3e-3", "3e-3"), ("_5e-3", "5e-3"), ("_1e-2", "1e-2")]
# methods = [("one_gauss", "1G", 1, 0), ("1G_plus", "1G+", 1, 1), ("1G_plus_plus", "1G++", 1, 2), ("1G_plus_plus_plus", "1G+++", 1, 3)]
# methods = [("default", 64, 64, 64, "Baseline", 0, "black"), ("default_tune", 32, 64, 16, "0G tune", 0, "maroon"), ("one_gauss", 64, 64, 64, "1G", 1, "slateblue"), ("one_gauss_tune", 64, 32, 64, "1G tune", 1, "cornflowerblue"), ("two_gauss", 64, 64, 64, "2G", 2, "seagreen"), ("two_gauss_tune", 4, 153, 150, "2G tune", 2, "turquoise")]
# methods = [("default_tune_l2_init", 0, 32, 65, 79), ("1G_tune_l2_init", 1, 159, 16, 217), ("2G_tune_l2_init", 2, 29, 207, 159), ("3G_tune_l2_init", 3, 80, 78, 221), ("5G_tune_l2_init", 5, 244, 234 ,28), ("7G_tune_l2_init", 7, 22, 201, 224), ("10G_tune_l2_init", 10, 4, 221, 202)]
# methods = [("1G_plus", 1, 1, "midnightblue"), ("2G_plus", 1, 2, "maroon"), ("3G_plus", 1, 3, "mediumturquoise")]
# methods = [("one_gauss", "1G", 1, 0, "black"), ("two_gauss", "2G", 2, 0, "firebrick"), ("three_gauss", "3G", 3, 0, "darkseagreen"), ("1G_plus", "1G+", 1, 1, "midnightblue"), ("1G_plus_plus", "1G++", 1, 2, "slateblue"), ("1G_plus_plus", "1G+++", 1, 3, "cornflowerblue")]
# methods = [("one_gauss", "1G", 1), ("1G_test1", "1G (test 1)", 1), ("1G_test2", "1G (test 2)", 1), ("1G_test3", "1G (test 3)", 1), ("1G_test4", "1G (test 4)", 1), ("1G_test5", "1G (test 5)", 1)]
# methods = [("one_gauss", "1G", 1), ("1G_006", "1G 0.6%", 1), ("1G_007", "1G 0.7%", 1), ("1G_009", "1G 0.9%", 1), ("1G_015", "1G 1.5%", 1), ("1G_020", "1G 2.0%", 1), ("1G_030", "1G 3.0%", 1)]
# methods = [("2G_8batch", "batch 8"), ("2G_16batch", "batch 16"), ("2G_32batch", "batch 32"), ("2G_64batch", "batch 64"), ("two_gauss", "batch 128"), ("2G_1280batch", "batch 1280"), ("2G_fullbatch", "full dataset")]
# methods = [("default", "Vanilla", 0), ("default_sgd", "0G SGD", 0), ("default_sgd_momentum", "0G SGD with momentum", 0),
#            ("two_gauss", "2G", 2), ("2G_sgd", "2G SGD", 2), ("2G_sgd_momentum", "2G SGD with momentum", 2)]
# methods = [("006", "0.6% S/B", 1), ("007", "0.7% S/B", 1), ("009", "0.9% S/B", 1), ("015", "1.5% S/B", 1), ("020", "2.0% S/B", 1),
#            ("030", "3.0% S/B", 1), ("040", "4.0% S/B", 1), ("050", "5.0% S/B", 1), ("100", "10.0% S/B", 1), ("150", "15.0% S/B", 1), ("250", "25.0% S/B", 1)]
# methods = [("two_gauss", "2G", 2, 64), ("2G_2e-3lr_512nodes", "2G 2e-3 lr 512 nodes", 2, 512), ("2Gcombobg_vs_bg", "2G 2e-3 lr 512 nodes BG", 2, 512)]

# normal plot
colors = cm.viridis(np.linspace(0., 0.95, len(methods)))

plt.figure(figsize=(8, 6))

model_evaluation.evaluate_tpr_fpr(device,"baseline_signal_vs_bg/", gauss_num = 0, signal_vs_bg=True, label = "Fully supervised", color = "black")
model_evaluation.evaluate_tpr_fpr(device,"EPiC_signal_vs_bg/", gauss_num = 0, use_EPiC=True, signal_vs_bg=True, label = "EPiC fully supervised", with_random=True, color = "green")

# model_evaluation.evaluate_tpr_fpr(device, "default_signal_vs_bg_hingeloss/", gauss_num=0, signal_vs_bg=True, hingeloss=True, label="0G fully supervised", color = "black")
# model_evaluation.evaluate_tpr_fpr(device, "2G_signal_vs_bg_hingeloss/", gauss_num=2, hingeloss=True, signal_vs_bg=True, label="2G fully supervised", color = "orange")
# model_evaluation.evaluate_tpr_fpr(device, "default_hingeloss/", gauss_num=0, hingeloss=True, label="0G", color = "blue")
# model_evaluation.evaluate_tpr_fpr(device, "2G_hingeloss/", gauss_num=2, hingeloss=True, label="2G", color = "teal", with_random=True)

# model_evaluation.evaluate_tpr_fpr(device, "two_gauss/", gauss_num=2, label="2G", color = "maroon")
# model_evaluation.evaluate_tpr_fpr(device, "2G_16batch_2e-3lr_512nodes/", l1=512,l2=512,l3=512, gauss_num=2, label="2G batch 16 2e-3 LR 512 nodes", color = "maroon")
# model_evaluation.evaluate_tpr_fpr(device, "2G_16batch_2e-3lr/", gauss_num=2, label="2G batch 16 2e-3 LR", color = "orange")
# model_evaluation.evaluate_tpr_fpr(device, "2G_16batch_512nodes/", l1=512,l2=512,l3=512, gauss_num=2, label="2G batch 16 512 nodes", color = "blue")
# model_evaluation.evaluate_tpr_fpr(device, "2G_2e-3lr_512nodes/",l1=512,l2=512,l3=512, gauss_num=2, label="2G combo", color = "green", with_random=True)

# model_evaluation.evaluate_tpr_fpr(device, "2G_16batch_2e-3lr_512nodes/", l1=512,l2=512,l3=512, gauss_num=2, label="2G batch 16, 2e-3LR, 512 nodes", color = "blue", with_random=True)

# count = 0
# for method, label, gauss in methods:
#     node = 64
#     dir = method + "/"
#     # gauss = 2
#     signal_vs_bg = False
#     with_random = False
#     l1 = node
#     l2 = node
#     l3 = node
#     if count == len(methods)-1:
#         with_random = True
#     model_evaluation.evaluate_tpr_fpr(device, dir, l1=l1, l2=l2,l3=l3, gauss_num = gauss, label = label, signal_vs_bg=signal_vs_bg, color = colors[count], with_random=with_random)
#     count += 1
plt.subplots_adjust(right=2.0)
plt.savefig("plots/results/roc_sic_epic.pdf", bbox_inches="tight")
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