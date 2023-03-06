from itertools import combinations
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import math
import matplotlib.pyplot as plt
import model_training


parser = argparse.ArgumentParser(
    description=("Configuring training parameters/features."), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--savedir', type=str, default="default/",
                    help='Path to directory where model and plots will be stored.')
parser.add_argument("--S_over_B", type=str, default="",
                    help="Signal vs background ratio")
parser.add_argument("--train_each_model", action="store_true", default=False,
                    help="retraining collapsed models")
parser.add_argument("--model_nums", action="store", type=str, nargs='*', default=["0", "1", "2", "3"],
                    help="model numbers for retraining, used when --train_each_model is activated")
parser.add_argument("--loss_per_batch", action="store_true", default=False,
                    help="loss vs batch's signal study - save no models")
parser.add_argument("--data_reduc", type=float, default=1.0,
                    help="duplicate the last feature")
parser.add_argument("--bg_vs_bg", action="store_true", default=False,
                    help="bg vs bg (data vs bg) task")
parser.add_argument("--num_model", type=int, default=10,
                    help="number of models (will be used in evaluation)")
parser.add_argument("--phi", action="store_true", default=False,
                    help="use phi")
parser.add_argument("--dropout", action="store_true", default=False,
                    help="use dropout")
parser.add_argument("--hingeloss", action="store_true", default=False,
                    help="use HingeEmbeddingLoss")
parser.add_argument("--sgd", action="store_true", default=False,
                    help="use SGD optimizer")
parser.add_argument("--momentum", type=float, default=0.0,
                    help="momentum of SGD optimizer, used when --sgd is used")
parser.add_argument("--morelayers", action="store_true", default=False,
                    help="use NN with 10 hidden layers")
parser.add_argument("--nodes", action="store", type=str, nargs='*', default=["64", "64", "64"],
                    help="number of nodes per layer")
parser.add_argument("--weight_seed", action="store_true", default=False,
                    help="Set random seed for weights")
parser.add_argument("--choose_features", action="store_true", default=False,
                    help="feature combination study")
parser.add_argument("--features", action="store", type=str, nargs='*', default=["0", "1", "2", "3"],
                    help="list of features in the feature study")
parser.add_argument("--gaussian", type=int, default=0,
                    help="number of gaussian input")
parser.add_argument("--uniform", type=int, default=0,
                    help="number of uniform input")
parser.add_argument("--duplicate_feature", type=int, default=0,
                    help="duplicate the last feature")
parser.add_argument("--epochs", type=int, default=100,
                    help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128,
                    help="number of batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3,
                    help="configuring learning rate")
parser.add_argument("--weight_decay", type=float, default=0,
                    help="configuring weight decay (regularization)")
parser.add_argument("--signal_vs_bg", action="store_true", default=False,
                    help="train over signal vs background")
args = parser.parse_args()

# if args.mjj:
#     print("Use mjj")
# else:
#     print("No mjj")

if args.signal_vs_bg:
    print("Signal vs Background.")
else:
    print("Data vs Bachkground.")

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def choose_feature(X1, X2, combination=[0,1,2,3]):
    new_X1 = np.ones((len(X1), 1))
    new_X2 = np.ones((len(X2), 1))
    for i in combination:
        new_X1 = np.concatenate((new_X1, X1[:, i].reshape((len(X1), 1))), axis = 1)
        new_X2 = np.concatenate((new_X2, X2[:, i].reshape((len(X2), 1))), axis=1)
    new_X1 = new_X1[:, 1:]
    new_X2 = new_X2[:, 1:]
    return new_X1, new_X2

##############################################################################################################################
# the actual training
if not os.path.exists('saved_model/' + args.savedir):
    os.mkdir('saved_model/' + args.savedir)
if not os.path.exists('plots/' + args.savedir):
    os.mkdir('plots/' + args.savedir)
print(args.savedir + " dir is created.")

l1, l2, l3 = int(args.nodes[0]), int(args.nodes[1]), int(args.nodes[2])
print("Training with " "l1, l2, l3: " + str(args.nodes) + ".")

X_train, X_val, y_train, y_val = model_training.data_loader(signal_vs_bg=args.signal_vs_bg, phi=args.phi, uniform_num=args.uniform, S_over_B=args.S_over_B)

if args.choose_features:
    print("Feature ranking study")
    combi = []
    for i in args.features:
        combi.append(int(i))
    print(combi)
    X_train, X_val = choose_feature(X_train, X_val, combi)

if args.batch_size == 0:
    batch_size = len(X_train)
else:
    batch_size = args.batch_size
    
epochs = args.epochs
learning_rate = args.learning_rate

if args.loss_per_batch:
    model_training.train_per_batch(args.savedir, device, X_train, X_val, y_train, y_val, l1=l1, l2=l2, l3=l3, use_SGD=args.sgd, momentum=args.momentum, morelayers=args.morelayers, hingeloss=args.hingeloss, dropout=args.dropout, num_model=args.num_model, signal_vs_bg=args.signal_vs_bg, data_reduc=args.data_reduc, phi=args.phi, gauss_num=args.gaussian, uniform_num=args.uniform, learning_rate=learning_rate, weight_decay=args.weight_decay, epochs=epochs, batch_size=batch_size)
elif args.train_each_model:
    collect_model_num = []
    for model_num in args.model_nums:
        collect_model_num.append(int(model_num))
    model_training.train_each_model(args.savedir, device, X_train, X_val, y_train, y_val, model_nums=collect_model_num, l1=l1, l2=l2, l3=l3, use_SGD=args.sgd, momentum=args.momentum, weight_seed=args.weight_seed, morelayers=args.morelayers, hingeloss=args.hingeloss, dropout=args.dropout, num_model=args.num_model, bg_vs_bg=args.bg_vs_bg, signal_vs_bg=args.signal_vs_bg, data_reduc=args.data_reduc, phi=args.phi, gauss_num=args.gaussian, uniform_num=args.uniform, learning_rate=learning_rate, weight_decay=args.weight_decay, epochs=epochs, batch_size=batch_size)
else:
    model_training.train_model(args.savedir, device, X_train, X_val, y_train, y_val, l1=l1, l2=l2, l3=l3, use_SGD=args.sgd, momentum=args.momentum, weight_seed=args.weight_seed, morelayers=args.morelayers, hingeloss=args.hingeloss, dropout=args.dropout, num_model=args.num_model, bg_vs_bg=args.bg_vs_bg, signal_vs_bg=args.signal_vs_bg, data_reduc=args.data_reduc, phi=args.phi, gauss_num=args.gaussian, uniform_num=args.uniform, learning_rate=learning_rate, weight_decay=args.weight_decay, epochs=epochs, batch_size=batch_size)