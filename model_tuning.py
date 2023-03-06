import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import numpy as np
import os
import math
import argparse
import sklearn
import define_model
import data_process
import model_training
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

parser = argparse.ArgumentParser(
    description=("How many gaussian inputs do you want?"), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gaussian", type = int, default = 0,
                    help="number of gaussian inputs")
args = parser.parse_args()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

batch_size = 128
print("Batch size = " + str(batch_size))

gaussian = args.gaussian
print("Number of guassian inputs is " + str(gaussian))

def train_model(config, checkpoint_dir=None, gauss_num=gaussian, epoch = 100):
    # data handling session
    X_train = np.load("/beegfs/desy/user/prangchp/repeat_manuel/X_files/X_train.npy")
    X_val = np.load("/beegfs/desy/user/prangchp/repeat_manuel/X_files/X_val.npy")
    y_train = np.load("/beegfs/desy/user/prangchp/repeat_manuel/X_files/y_train.npy")
    y_val = np.load("/beegfs/desy/user/prangchp/repeat_manuel/X_files/y_val.npy")
    print("Data is loaded.")
    X_train, y_train = model_training.data_reduction(X_train, y_train, data_reduc=1.0, signal_vs_bg=False)
    X_val, y_val = model_training.data_reduction(X_val, y_val, data_reduc=1.0, signal_vs_bg=False)

    # Class weight
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train[:, 0]),y=y_train[:, 0])
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    print("Class Weight: " + str(class_weights))

    # define X, y
    X_train = torch.from_numpy(X_train[:, :4 + gauss_num]).float().to(device)
    X_val = torch.from_numpy(X_val[:, :4 + gauss_num]).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)

    loss_fn = define_model.BCELoss_class_weighted(class_weights)

    # count number of inputs
    input_number = len(X_train[0])
    
    # redefining model everytime we start training new model to clear out the old graph
    model = define_model.NeuralNetwork(input_number, l1=64, l2=64, l3=64).to(device)
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=config["momentum"])
    print("Model and Optimizer are defined.")

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # prepare Dataset
    train_data = data_process.MakeASet(X_train, y_train)
    val_data = data_process.MakeASet(X_val, y_val)

    # make DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=len(val_data), shuffle=True)

    train_loss_coll = []

    size_train = len(train_loader.dataset)
    n_batch = math.ceil(size_train/batch_size)

    for t in range(epoch):
        # print(f"Epoch {t+1}\n-------------------------------")
        model.train()

        train_loss, val_loss, train_correct, val_correct = 0, 0, 0, 0
        train_loss_mean = 0

        # training
        for batch, (X, y) in enumerate(train_loader):
            # Compute prediction error
            pred = model(X)
            train_loss = loss_fn(pred, y)
            train_loss_mean += train_loss.mean().item()

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_loss_coll.append(train_loss_mean/n_batch)

        # validation
        model.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(val_loader):
                val_pred = model(X)
                val_loss = loss_fn(val_pred, y)
                val_correct = (val_pred.round() == y).sum().item() # Accuracy
                val_loss = val_loss.item()
                val_correct /= len(X) 
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=val_loss)
    print("Training is Done!")

def tune_it_bayes(num_samples=20, max_num_epochs=10, grace_period = 1, gpus_per_trial=1):
    space = {
        "momentum": (0,0.9)
    }
    initialize = [{
        "momentum": 0.9
    }]
    bayesopt = BayesOptSearch(space=space, metric="loss", mode="min", points_to_evaluate=initialize)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]) 
    result = tune.run(
        train_model,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        num_samples=num_samples,
        search_alg=bayesopt,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config for " + str(gaussian) + " gauss inputs: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

tune_it_bayes(num_samples=150, max_num_epochs=50, grace_period = 20, gpus_per_trial=0.2)
