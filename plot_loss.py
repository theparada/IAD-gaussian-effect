import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import combinations

def get_loss(dir, for_train = False):
    losses = []
    for model_num in range(10):
        if for_train:
            losses.append(np.load("saved_model/"+dir +
                          "train_losses_model_"+str(model_num+1)+".npy"))
        else:
            losses.append(np.load("saved_model/"+dir+"val_losses_model_"+str(model_num+1)+".npy"))
    losses = np.stack(losses)
    losses = (losses - 0.69)*1e3
    return losses

def make_loss(dir, for_train = False):
    dir = dir + "/"
    matrix = get_loss(dir, for_train=for_train)
    median = np.median(matrix, axis = 0)
    std = (np.nanquantile(matrix, 0.16, axis=0), np.nanquantile(matrix, 0.84, axis=0))
    return matrix, median, std
    
def plot_loss(dir, label, val_color, train_color, with_train = False, epoch=100):
    val_matrix, val_median, val_std = make_loss(dir)
    if with_train:
        train_matrix, train_median, train_std = make_loss(dir, for_train=True)
        plt.title("Loss")

        plt.plot(range(epoch), train_median[:epoch], color = train_color, label = label)
        plt.fill_between(range(epoch), train_std[0][:epoch], train_std[1][:epoch], color = train_color, alpha=0.3)
    else:
        plt.title("Validation Loss")

    plt.plot(range(epoch), val_median[:epoch],
                color=val_color, label=label)
    plt.fill_between(
        range(epoch), val_std[0][:epoch], val_std[1][:epoch], color=val_color, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.xlim([0, 50])
    plt.ylim([2.6, 5.0])
    plt.ylabel('Binary Cross Entropy Loss', fontsize=12)
    plt.legend(loc="upper left")


# models = ["default", "default_l2", "default_tune", "default_tune_L2_en10", "one_gauss", "1G_l2", "one_gauss_tune", "1G_tune_L2_en10_realone", "two_gauss", "2G_l2", "two_gauss_tune"]

# for model in models:
#     plot_loss(model)

# methods = [("default", "Vanilla"), ("one_gauss", "1G"), ("two_gauss", "2G"), ("three_gauss", "3G"), ("five_gauss", "5G"), ("seven_gauss", "7G"), ("ten_gauss", "10G")]

# # methods = [("", "100% data", 1), ("080", "80% data", 1), ("050", "50% data", 1), ("030","30% data", 1), ("020", "20% data", 1), ("015", "15% data", 1), ("010", "10% data", 1)]
# # sbs = ["006", "007", "009", "015", "020", "030", "040", "050", "100", "150", "250"]

# methods = [("default", "Vanilla", "black"),("0G_2e-3lr_512nodes", "0G combo", "blue"), ("two_gauss", "2G", "maroon"), ("2G_2e-3lr_512nodes", "2G combo", "green")]

methods = [("two_gauss", "2G", 2, 64), ("2G_2e-3lr_512nodes", "2G 2e-3 lr 512 nodes", 2, 512), ("2Gcombobg_vs_bg", "2G 2e-3 lr 512 nodes BG", 2, 512)]

colors = cm.viridis(np.linspace(0., 0.95, len(methods)))

plt.figure(figsize=(11, 7))
count = 0
for (method, label, gauss, node) in methods:
    dir = method + "/"
    plot_loss(dir=dir, label=label, val_color=colors[count], train_color="m", epoch=100)
    count += 1
plt.savefig("all_plots/loss_combo.pdf")
plt.close()