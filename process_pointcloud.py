import pandas as pd
import numpy as np
import os
from os.path import join
import h5py
import argparse

parser = argparse.ArgumentParser(
    description=("Prepare LHCO dataset."),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# use the default value
parser.add_argument("--S_over_B", type=float, default=0.006361658645922605,
                    help="Signal over background ratio in the signal region.")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed for the mixing")
parser.add_argument("--outdir", type=str, default="vanilla_data/",
                    help="output directory")
args = parser.parse_args()

# load data
s_sr = h5py.File("data/LHCO_pointcloud/events_s_sr.h5")
s_br = h5py.File("data/LHCO_pointcloud/events_s_br.h5")
b_sr = h5py.File("data/LHCO_pointcloud/events_b_sr.h5")
b_br = h5py.File("data/LHCO_pointcloud/events_b_br.h5")
ex_bg1 = h5py.File("data/LHCO_pointcloud/extra_bg1.h5")
ex_bg2 = h5py.File("data/LHCO_pointcloud/extra_bg2.h5")

def prepare_jets(data, mjjmin=3.3, mjjmax=3.7):
    # make np array
    prejet1 = np.array(data["jet1_PFCands"])
    m_prej1 = np.array(data["jet_kinematics"][:, 5])
    prejet2 = np.array(data["jet2_PFCands"])
    m_prej2 = np.array(data["jet_kinematics"][:, 9])
    label = np.array(data["truth_label"])
    mjj = np.array(data["jet_kinematics"][:, 0])
    mjj = mjj/1000
    # mjj = np.resize(mjj, (len(mjj), 1))

    # reshape to (, 200 , 3) 
    prejet1 = np.resize(prejet1, (len(prejet1), 200, 3))
    prejet2 = np.resize(prejet2, (len(prejet2), 200, 3))

    # sort jets
    for i in range(len(prejet1)):
        hold = np.array([])
        if (m_prej1[i]<m_prej2[i]):
            hold = prejet1[i]
            prejet1[i] = prejet2[i]
            prejet2[i] = hold

    # make jet labels
    jet1 = np.concatenate((prejet1, np.zeros((len(prejet1), 200, 1))), axis = 2)    
    jet2 = np.concatenate((prejet2, np.ones((len(prejet2), 200, 1))), axis = 2)
    
    full_events = np.concatenate((jet1, jet2), axis=1)

    # mask mjj
    mask = (mjj>mjjmin) & (mjj<mjjmax)
    full_events = full_events[mask]
    label = label[mask]
    return full_events, label

def combine_data(data1, data2):
    x1, y1 = prepare_jets(data1)
    x2, y2 = prepare_jets(data2)
    features = np.concatenate((x1, x2), axis=0)
    label = np.concatenate((y1, y2), axis=0)
    return features, label

# prepare to make datasets
sig, label_sig = combine_data(s_sr, s_br)
bkg, label_bkg = combine_data(b_sr, b_br)
extra1, label_extra1 = prepare_jets(ex_bg1)
extra2, label_extra2 = prepare_jets(ex_bg2)

# determine sig in S/B
n_sig = int(args.S_over_B*len(sig))
n_sig_test = 20000

# split training and validation sets
data_all = np.concatenate((bkg, sig[:n_sig]), axis=0)
label_all = np.concatenate((label_bkg, label_sig[:n_sig]), axis=0)
indices_all = np.array(range(len(data_all))).astype('int')
np.random.shuffle(indices_all)
data_all = data_all[indices_all]
label_all = label_all[indices_all]

X_train_data = data_all[:60000]
X_val_data = data_all[60000:120000]
y_train_data = label_all[:60000]
y_val_data = label_all[60000:120000]

# test signals
n_sig_test = 20000
X_test_sig = sig[:n_sig_test]
y_test_sig = label_sig[:n_sig_test]

# train and val extra signals for signal vs bg task
n_extrasig_train =  (sig.shape[0]-n_sig_test)//2

X_train_extrasig = sig[n_sig_test:n_sig_test+n_extrasig_train]
X_val_extrasig = sig[n_sig_test+n_extrasig_train:]
y_train_extrasig = label_sig[n_sig_test:n_sig_test+n_extrasig_train]
y_val_extrasig = label_sig[n_sig_test+n_extrasig_train:]

# splitting extra bkg (1) into train, val and test set
n_bkg_test = 40000
X_test_bg = extra1[:n_bkg_test]
y_test_bg = label_extra1[:n_bkg_test]

# train and val backgrounds
n_extrabkg_train =  (extra1.shape[0]-n_bkg_test)//2

X_train_bg = extra1[n_bkg_test:n_bkg_test+n_extrabkg_train]
X_val_bg = extra1[n_bkg_test+n_extrabkg_train:]
y_train_bg = label_extra1[n_bkg_test:n_bkg_test+n_extrabkg_train]
y_val_bg = label_extra1[n_bkg_test+n_extrabkg_train:]

# ready to use datasets
X_train = np.concatenate((X_train_data, X_train_bg), axis=0)
X_val = np.concatenate((X_val_data, X_val_bg), axis=0)
X_test = np.concatenate((X_test_sig, X_test_bg, extra2), axis=0)

y_train = np.concatenate((y_train_data, y_train_bg), axis=0)
y_val = np.concatenate((y_val_data, y_val_bg), axis=0)
y_test = np.concatenate((y_test_sig, y_test_bg, label_extra2), axis=0)