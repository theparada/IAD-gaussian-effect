import numpy as np
from torch.utils.data import Dataset

# data preprocessing
def data_normalize(x, pre_x):
    x = (x - pre_x.mean(axis=0))/pre_x.std(axis = 0)
    print("Normalized X")
    return x

def data_normalize_EPiC(x, pre_x):
    for i in range(len(x[0,0])):
        x_mean = np.mean(np.ndarray.flatten(pre_x[:,:,i]))
        x_std = np.std(np.ndarray.flatten(pre_x[:,:,i]))
        x[:,:,i] = (x[:,:,i] - x_mean)/x_std
    return x

def normalize_datasets(x1, x2, pre_x):
    x1 = data_normalize(x1, pre_x)
    x2 = data_normalize(x2, pre_x)
    return x1, x2

# create gaussian noise
def add_gauss(x):
    gauss = np.reshape(np.random.normal(0, 1, len(x)), (len(x), 1))
    x = np.concatenate((x, gauss), axis = 1)
    return x

# insert gaussian
def make_multiple_gauss(x1, x2, x3, num):
    i = 0
    while i<num:
        x1 = add_gauss(x1)
        x2 = add_gauss(x2)
        x3 = add_gauss(x3)
        i += 1
        print("Gaussian " + str(i) + " included.")
    return x1, x2, x3

def uniform_noise(x1, x2, x3, num):
    i = 0
    while i<num:
        x1 = np.concatenate((x1, np.random.uniform(0.0, 1.0, (len(x1), 1))), axis = 1)
        x2 = np.concatenate((x2, np.random.uniform(0.0, 1.0, (len(x2), 1))), axis = 1)
        x3 = np.concatenate((x3, np.random.uniform(0.0, 1.0, (len(x3), 1))), axis = 1)
        i += 1
    return x1, x2, x3

# duplicate a feature (test whether the duplication affects the performance)
def duplicate_feature(x, input_num):
    if input_num>0:
        for i in range(input_num):
            x = np.concatenate((x, x[:, -1].reshape((len(x), 1))), axis=1)
    return x

# include constant feature
def constant_feature(x):
    constant = np.ones((len(x), 1))
    x = np.concatenate((x, constant), axis = 1)
    return x

# define X and y
def make_features(data, use_mjj=False, for_test = False):
    # for X
    if use_mjj:
        X = data[:, :len(data[0]) - 2]
    else:
        X = data[:, 1:len(data[0]) - 2]
        
    if for_test:
        y = np.reshape(data[:, -2], (len(data), 1)) # signal vs bg only for test set
    else:
        y = np.reshape(data[:, -2:], (len(data), 2)) # signal vs bg and data vs bg
    return X, y

class MakeASet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

    def getsize(self):
        return len(self.X), len(self.X[0])

# Download data
def data_loader(dir = "new_separated_data/", signal_vs_bg = False, train = False, val = False):
    if signal_vs_bg:
        # for signal vs bg training, we need to saturate the signals for better performance
        if train:
            load_data = np.load(dir + 'innerdata_train.npy')
            load_bg = np.load(dir + 'innerdata_extrabkg_train.npy')
            extra_sig = np.load(dir + 'innerdata_extrasig_train.npy')

            load_data = np.concatenate((load_data, np.ones((len(load_data), 1))), axis=1)
            load_bg = np.concatenate((load_bg, np.zeros((len(load_bg), 1))), axis=1)
            extra_sig = np.concatenate((extra_sig, np.ones((len(extra_sig), 1))), axis=1)

            data = np.concatenate((load_data, extra_sig, load_bg), axis = 0)
        elif val:
            load_data = np.load(dir + 'innerdata_val.npy')
            load_bg = np.load(dir + 'innerdata_extrabkg_val.npy')
            extra_sig = np.load(dir + 'innerdata_extrasig_val.npy')

            load_data = np.concatenate((load_data, np.ones((len(load_data), 1))), axis=1)
            load_bg = np.concatenate((load_bg, np.zeros((len(load_bg), 1))), axis=1)
            extra_sig = np.concatenate((extra_sig, np.ones((len(extra_sig), 1))), axis=1)
            data = np.concatenate((load_data, extra_sig, load_bg), axis = 0)
    else:
        if train:
            load_data = np.load(dir + 'innerdata_train.npy')
            load_bg = np.load(dir + 'innerdata_extrabkg_train.npy')
        elif val:
            load_data = np.load(dir + 'innerdata_val.npy')
            load_bg = np.load(dir + 'innerdata_extrabkg_val.npy')
        else:
            load_data = np.load(dir + 'innerdata_test.npy')
            load_bg = np.load(dir + 'innerdata_extrabkg_test.npy')

        # labeling data vs bg
        load_data = np.concatenate((load_data, np.ones((len(load_data), 1))), axis=1)
        load_bg = np.concatenate((load_bg, np.zeros((len(load_bg), 1))), axis=1)

        data = np.concatenate((load_data, load_bg), axis = 0)
        np.random.shuffle(data)
    return data

def data_prep(data_train, data_val, data_test, gauss_num = 0, uniform_num = 0):
    # make X and y
    X_train, y_train = make_features(data_train, for_test=False)
    X_val, y_val = make_features(data_val, for_test=False)
    X_test, y_test = make_features(data_test, for_test=True)

    # include Gaussian noise
    if gauss_num > 0:
        X_train, X_val, X_test = make_multiple_gauss(X_train, X_val, X_test, gauss_num)
    if uniform_num>0:
        X_train, X_val, X_test = uniform_noise(X_train, X_val, X_test, uniform_num)
    print("X_files will be normalized before being used.")
    return X_train, X_val, X_test, y_train, y_val, y_test 