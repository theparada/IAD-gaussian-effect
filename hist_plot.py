import numpy as np
import matplotlib.pyplot as plt
import data_process
import pointcloud_loader

kwargs = dict(histtype='stepfilled', alpha=0.3, bins=80)

def hist_plot(data):
    plt.hist(data, label="event", **kwargs)
    plt.legend(loc = 'upper right')
    plt.xlabel("y")
    plt.ylabel("Events")
    plt.savefig("X_files/phi_hist.png")
    plt.close()


def separate_sig(x, y):
    x_sig = []
    x_bg = []
    count = 0
    for i in y[:, 0]:
        if i == 1:
            x_sig.append(x[count])
            count += 1
        else:
            x_bg.append(x[count])
            count += 1
    return x_sig, x_bg

# data_train = data_process.data_loader("new_separated_data/", train=True)

# X_sb = np.load("X_files/X_train_sb.npy")
# X_db = np.load("X_files/X_test.npy")

def data_vs_bg_label(x, y, sig_num, len_train):
    x_sig = []
    x_bg = []
    y_sig = []
    y_bg = []
    count = 0
    for i in y:
        if i == 1:
            x_sig.append(x[count])
            y_sig.append(y[count])
            count += 1
        else:
            x_bg.append(x[count])
            y_bg.append(y[count])
            count += 1 
    x_data = np.concatenate((x_sig[:sig_num], x_bg[:(60000 - sig_num)]), axis = 0)
    x_bg_bg = x_bg[60001 - sig_num: len_train - sig_num]
    y_data = np.concatenate((y_sig[:sig_num], y_bg[:(60000 - sig_num)]), axis = 0)
    y_bg_bg = y_bg[60001 - sig_num: len_train - sig_num]
    new_x = np.concatenate((x_data, x_bg_bg), axis=0)
    new_y = np.concatenate((y_data, y_bg_bg), axis=0)
    return new_x, new_y

def data_normalize_EPiC(x, x_mean, x_std):
    for i in range(len(x[0])):
        if i != 3:
            x[...,i] = (x[...,i] - x_mean[i])/x_std[i]
    return x


X_train = np.load("data/X_files_EPiC/X_train.npy")
y_train = np.load("data/X_files_EPiC/y_train.npy")
train_data = pointcloud_loader.PointCloudDataset_ZeroPadded(X_train, y_train)
mean, std = train_data.get_means_stds()

X_val = np.load("data/X_files_EPiC/X_val.npy")
y_val = np.load("data/X_files_EPiC/y_val.npy")
val_mask = (X_val[...,0] != 0.)
val_mask = val_mask.reshape(val_mask.shape[0], val_mask.shape[1], 1)
X_val = np.where(val_mask, X_val, np.nan)
X_val = data_process.data_normalize_EPiC(X_val, mean, std)
sig_num = len(y_val[y_val==1]) - 60000
print(sig_num)
X_test = np.load("data/X_files_EPiC/X_test.npy")
y_test = np.load("data/X_files_EPiC/y_test.npy")
test_mask = (X_test[...,0] != 0.)
test_mask = test_mask.reshape(test_mask.shape[0], test_mask.shape[1], 1)
X_test = np.where(test_mask, X_test, np.nan)
X_test = data_process.data_normalize_EPiC(X_test, mean, std)

X_sb, y_sb = data_vs_bg_label(X_test, y_test, sig_num, len(y_val))

features = ["$p_T$ (GeV)", "$\\eta$", "$\\phi$"]

val_sig = []
val_bg = []
for i in range(len(y_val)):
    if y_val[i, 0] == 1:
        val_sig.append(X_val[i])
    else:
        val_bg.append(X_val[i])
val_sig = np.vstack(val_sig)
val_bg = np.vstack(val_bg)

test_sig = []
test_bg = []
for i in range(len(y_sb)):
    if y_sb[i, 0] == 1:
        test_sig.append(X_sb[i])
    else:
        test_bg.append(X_sb[i])
test_sig = np.vstack(test_sig)
test_bg = np.vstack(test_bg)

for i in range(len(features)):
    kwargs1 = dict(histtype='stepfilled', alpha=0.3, bins=80)
    kwargs2 = dict(histtype='step', alpha=0.3, bins=80)
    plt.hist(val_bg[..., i].flatten(), range = (np.nanmin(val_bg[..., i].flatten()), np.nanmax(val_bg[..., i].flatten())), label="val background", **kwargs1)
    plt.hist(val_sig[..., i].flatten(), range= (np.nanmin(val_bg[..., i].flatten()), np.nanmax(val_bg[..., i].flatten())), label="val signal", **kwargs1)
    plt.hist(test_bg[..., i].flatten(), range = (np.nanmin(val_bg[..., i].flatten()), np.nanmax(val_bg[..., i].flatten())), label="test background", **kwargs2)
    plt.hist(test_sig[..., i].flatten(), range= (np.nanmin(val_bg[..., i].flatten()), np.nanmax(val_bg[..., i].flatten())), label="test signal", **kwargs2)
    plt.legend(loc = "upper right", frameon=False, fontsize=11)
    plt.xlabel(features[i], fontsize=11)
    plt.ylabel("events", fontsize=11)
    plt.yscale("log")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig("plots/results/featureepic"+str(i)+".png")
    plt.show()
    plt.close()

# sig, bg = separate_sig(X_train, y_train)
# print(len(sig))
# sig = np.reshape(sig, (len(sig), len(sig[0])))
# bg = np.reshape(bg, (len(bg), len(bg[0])))

# range = (-0.1, 1.)

# plt.hist(bg[:, 0], label="Background", range=range, ** kwargs)
# plt.hist(sig[:, 0], label="Signal", range=range,  **kwargs)
# plt.legend(loc = 'upper right')
# plt.yscale('log')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("value", fontsize=12)
# plt.ylabel("Events", fontsize=12)
# plt.savefig("all_plots/feature0_sb_debug.pdf")
# plt.close()