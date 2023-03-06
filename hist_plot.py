import numpy as np
import matplotlib.pyplot as plt
import data_process

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

X_sb = np.load("X_files/X_train_sb.npy")
X_db = np.load("X_files/X_test.npy")

print(len(X_sb[0]))
print(len(X_db[0]))

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