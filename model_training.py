import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import math
import matplotlib.pyplot as plt
import sklearn
import define_model
import data_process

def data_reduction(x, y, data_reduc, bg_vs_bg=False, signal_vs_bg=False):
    x_sig = []
    x_bg = []
    y_sig = []
    y_bg = []
    count = 0
    for i in y[:, 0]:
        if i == 1:
            x_sig.append(x[count])
            if signal_vs_bg:
                y_sig.append(y[count, 0])   
            else: 
                y_sig.append(y[count, 1])
            count += 1
        else:
            x_bg.append(x[count])
            if signal_vs_bg:
                y_bg.append(y[count, 0])
            else:
                y_bg.append(y[count, 1])
            count += 1
    np.stack(x_sig)
    np.stack(x_bg)
    np.stack(y_sig)
    np.stack(y_bg)
    if bg_vs_bg:
        # for bk (data) vs bg (bg) task
        new_x = np.array(x_bg[:round(data_reduc*len(x_bg))])
        new_y = y_bg[:round(data_reduc*len(y_bg))]
        new_y = np.reshape(new_y, (len(new_y), 1))
    else:
        new_x = np.concatenate((x_sig[:round(data_reduc*len(x_sig))], x_bg[:round(data_reduc*len(x_bg))]), axis = 0)
        new_y = np.concatenate((y_sig[:round(data_reduc*len(y_sig))], y_bg[:round(data_reduc*len(y_bg))]), axis = 0)
        new_y = np.reshape(new_y, (len(new_y), 1))
    return new_x, new_y

def data_loader(signal_vs_bg=False, phi=False, uniform_num=0, S_over_B=""):
    # y's are 190k*2 matrices where y[:, 0] and y[:, 1] are "sig vs bg" and "data vs bg" respectively
    if signal_vs_bg:
        X_train = np.load("X_files/X_train_sb.npy")
        X_val = np.load("X_files/X_val_sb.npy")

        y_train = np.load("X_files/y_train_sb.npy")
        y_val = np.load("X_files/y_val_sb.npy")
    else:
        if phi:
            X_train = np.load("X_files/X_train_phi.npy")
            X_val = np.load("X_files/X_val_phi.npy")

            y_train = np.load("X_files/y_train_phi.npy")
            y_val = np.load("X_files/y_val_phi.npy")
        elif uniform_num > 0:
            X_train = np.load("X_files/X_train_uni.npy")
            X_val = np.load("X_files/X_val_uni.npy")

            y_train = np.load("X_files/y_train_uni.npy")
            y_val = np.load("X_files/y_val_uni.npy")
        else:
            X_train = np.load("X_files/X_train"+S_over_B+".npy")
            X_val = np.load("X_files/X_val"+S_over_B+".npy")

            y_train = np.load("X_files/y_train"+S_over_B+".npy")
            y_val = np.load("X_files/y_val"+S_over_B+".npy")
    return X_train, X_val, y_train, y_val

def train_model(save_dir, device, pre_X_train, X_val, y_train, y_val, l1 = 64, l2 = 64, l3 = 64, morelayers=False, dropout=False, hingeloss=False, use_SGD = False, momentum=0, num_model = 10, bg_vs_bg=False, signal_vs_bg=False, weight_seed=False, phi=False, data_reduc=1.0, uniform_num=0, gauss_num=0, weight_decay=0., learning_rate=1e-3, epochs=100, batch_size=128):
    pre_X_train, y_train, _ = data_reduction(pre_X_train, y_train, data_reduc=data_reduc, bg_vs_bg=bg_vs_bg, signal_vs_bg=signal_vs_bg)
    X_val, y_val, _ = data_reduction(X_val, y_val, data_reduc=1.0, bg_vs_bg=bg_vs_bg, signal_vs_bg=signal_vs_bg)
    # y_train is now 190k*1 matrix 
    # normalize datasets here because raw X_train is needed in the evaluation 
    X_train, X_val = data_process.normalize_datasets(pre_X_train, X_val, pre_X_train)
    print("X_train and X_val are normalized.")

    # Class weight
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train[:, 0]), y=y_train[:, 0])
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    print("Class Weight: " + str(class_weights))
    
    if hingeloss:
        loss_fn = torch.nn.HingeEmbeddingLoss()
        y_train[y_train==0]=-1
        y_val[y_val==0]=-1
    else:
        loss_fn = define_model.BCELoss_class_weighted(class_weights)

    # define X, y
    if phi:
        X_train = torch.from_numpy(X_train[:, :5]).float().to(device)
        X_val = torch.from_numpy(X_val[:, :5]).float().to(device)
    elif uniform_num>0:
        X_train = torch.from_numpy(X_train[:, :4 + uniform_num]).float().to(device)
        X_val = torch.from_numpy(X_val[:, :4 + uniform_num]).float().to(device)
    else:
        X_train = torch.from_numpy(X_train[:, :4 + gauss_num]).float().to(device)
        X_val = torch.from_numpy(X_val[:, :4 + gauss_num]).float().to(device)

    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)

    # count number of inputs
    input_number = len(X_train[0])
    print("Input Number is " + str(input_number) + " nodes.")

    # prepare Dataset
    train_data = data_process.MakeASet(X_train, y_train)
    val_data = data_process.MakeASet(X_val, y_val)

    # make DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=len(val_data), shuffle=True)

    for model_num in range(num_model):
        print(f"Model {model_num+1}\n-------------------------------")
        # redefining model everytime we start training new model to clear out the old graph
        if weight_seed:
            torch.manual_seed(19960609*model_num)

        if dropout:
            model = define_model.NeuralNetworkDropout(input_number, l1, l2, l3).to(device)
        elif morelayers:
            model = define_model.NeuralNetworkMoreLayers(input_number, nodes=l1).to(device)
        else:    
            model = define_model.NeuralNetwork(input_number, l1, l2, l3).to(device)
       
        if use_SGD:
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            print("use SGD optimizer")
        else:
            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            print("use Adam optimizer")
        
        train_loss_coll = []
        val_loss_coll = []

        # these change with data reduction
        size_train = len(train_loader.dataset) 
        n_batch = math.ceil(size_train/batch_size)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            model.train()
            train_loss, val_loss, train_loss_mean = 0, 0, 0

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
                
                # train_correct = (pred.round() == y).sum().item()/batch_size

            print(f"Training loss: {train_loss_mean/n_batch:>7f}")

            train_loss_coll.append(train_loss_mean/n_batch)

            # validation
            model.eval()
            # val_loss_mean = 0
            with torch.no_grad():
                for batch, (X, y) in enumerate(val_loader):
                    val_pred = model(X)
                    val_loss = loss_fn(val_pred, y)
                    # val_loss_mean += val_loss.mean().item()
                    # val_correct = (val_pred.round() == y).sum().item()
                    # val_correct /= len(X)

            # val_correct = val_correct/math.ceil(len(X_val)/batch_size)
            # val_loss_mean = val_loss_mean/math.ceil(len(X_val)/batch_size)
            print(f"Validation loss: {val_loss:>8f} \n")

            val_loss_coll.append(val_loss.item())
            # val_corr_coll.append(val_correct)

            # save model each epoch
            np.save(os.path.join('saved_model/' + save_dir, "train_losses_model_"+str(model_num+1)+".npy"), train_loss_coll)
            np.save(os.path.join('saved_model/'+ save_dir, "val_losses_model_"+str(model_num+1)+".npy"), val_loss_coll)
            torch.save(model.state_dict(), os.path.join('saved_model/' + save_dir, 'saved_training_model'+str(model_num+1)+'_epoch'+str(t+1)+'.pt'))

        print("Model "+str(model_num+1)+" training is Done!")

        # Loss values plot
        plt.plot(range(epochs), train_loss_coll, 'm.-', label = "average train loss")
        plt.plot(range(epochs), val_loss_coll, 'c.-', label = "val loss")
        plt.xlabel('epoch')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.legend(loc="upper left")
        plt.savefig('all_plots/' + save_dir + 'train_loss_value_model'+str(model_num+1)+'.png')
        plt.close()
        print("Finish plotting loss functions.")

def train_per_batch(save_dir, device, pre_X_train, X_val, y_train, y_val, l1 = 64, l2 = 64, l3 = 64, morelayers=False, dropout=False, hingeloss=False, use_SGD = False, momentum=0, num_model = 10, signal_vs_bg=False, phi=False, data_reduc=1.0, uniform_num=0, gauss_num=0, weight_decay=0., learning_rate=1e-3, epochs=100, batch_size=128):
    # normalize datasets here because raw X_train is needed in the evaluation 
    X_train, X_val = data_process.normalize_datasets(pre_X_train, X_val, pre_X_train)
    print("X_train and X_val are normalized.")
    
    # Class weight
    if signal_vs_bg:
        class_weights = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train[:, 0]), y=y_train[:, 0])
    else:
        class_weights = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train[:, 1]), y=y_train[:, 1])
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    print("Class Weight: " + str(class_weights))

    if hingeloss:
        loss_fn = torch.nn.HingeEmbeddingLoss()
        y_train[y_train==0]=-1
        y_val[y_val==0]=-1
    else:
        loss_fn = define_model.BCELoss_class_weighted(class_weights)

    # define X, y
    if phi:
        X_train = torch.from_numpy(X_train[:, :5]).float().to(device)
        X_val = torch.from_numpy(X_val[:, :5]).float().to(device)
    elif uniform_num>0:
        X_train = torch.from_numpy(X_train[:, :4 + uniform_num]).float().to(device)
        X_val = torch.from_numpy(X_val[:, :4 + uniform_num]).float().to(device)
    else:
        X_train = torch.from_numpy(X_train[:, :4 + gauss_num]).float().to(device)
        X_val = torch.from_numpy(X_val[:, :4 + gauss_num]).float().to(device)

    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)

    # count number of inputs
    input_number = len(X_train[0])
    print("Input Number is " + str(input_number) + " nodes.")

    # prepare Dataset
    train_data = data_process.MakeASet(X_train, y_train)
    val_data = data_process.MakeASet(X_val, y_val)

    # make DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=len(val_data), shuffle=True)

    if dropout:
        model = define_model.NeuralNetworkDropout(input_number, l1, l2, l3).to(device)
    elif morelayers:
        model = define_model.NeuralNetworkMoreLayers(input_number, nodes=l1).to(device)
    else:    
        model = define_model.NeuralNetwork(input_number, l1, l2, l3).to(device)
    
    if use_SGD:
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        print("use SGD optimizer")
    else:
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print("use Adam optimizer")
    
    train_loss_coll = []
    val_loss_coll = []
    count_sig_coll = []

    # these change with data reduction

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        # training
        for batch, (X, y) in enumerate(train_loader):
            if signal_vs_bg:
                y = torch.reshape(y[:, 0], (len(y), 1))
                count_sig = 0
                for sig in y:
                    if sig == 1:
                        count_sig += 1
                count_sig_coll.append(count_sig)
            else:
                count_sig = 0
                for sig in y[:, 0]:
                    if sig == 1:
                        count_sig += 1
                count_sig_coll.append(count_sig)
                y = torch.reshape(y[:, 1], (len(y), 1))
            
            model.train()
            train_loss, val_loss = 0, 0

            # Compute prediction error
            pred = model(X)
            train_loss = loss_fn(pred, y)
            # print(f"Training loss: {train_loss:>7f}")

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_coll.append(train_loss.item())

            # validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                if signal_vs_bg:
                    val_loss = loss_fn(val_pred, torch.reshape(y_val[:, 0], (len(y_val), 1)))
                else:
                    val_loss = loss_fn(val_pred, torch.reshape(y_val[:, 1], (len(y_val), 1)))
                val_loss_coll.append(val_loss.item())
                # print(f"Validation loss: {val_loss:>7f}")

        val_loss_change = []
        for i in range(len(val_loss_coll)):
            if val_loss_coll[i] is not val_loss_coll[-1]:
                val_loss_change.append(val_loss_coll[i+1] - val_loss_coll[i])
        # sig_vs_loss = np.concatenate((np.reshape(count_sig_coll[1:], (len(count_sig_coll), 1)), np.reshape(val_loss_change, (len(val_loss_change), 1))), axis=0)
        sig_list = np.unique(count_sig_coll)

        plt.figure(figsize=(8, 5))
        for sig in sig_list:
            data_for_plot = []
            for num_sig, loss in zip(count_sig_coll[1:], val_loss_change):
                if num_sig == sig:
                    data_for_plot.append(loss)
            plt.hist(data_for_plot, histtype='step', bins=np.arange(-3e-4, 3e-4, 3.333333333e-5), range=(np.amin(val_loss_change), np.amax(val_loss_change),), label= str(sig) +" sigs, %.1e" %np.mean(data_for_plot) + " avg, %.1e" %np.std(data_for_plot)+" std")
            plt.legend(loc = 'upper left')
            plt.xlabel("Change of Validation Loss")
            plt.ylabel("Batches")
            plt.yscale('log')
            plt.savefig("all_plots/loss_vs_batch_"+str(gauss_num)+"G.pdf", bbox_inches="tight")
        plt.close()

def train_each_model(save_dir, device, pre_X_train, X_val, y_train, y_val, model_nums, l1 = 64, l2 = 64, l3 = 64, morelayers=False, dropout=False, hingeloss=False, use_SGD = False, momentum=0, num_model = 10, signal_vs_bg=False, weight_seed=False, phi=False, data_reduc=1.0, uniform_num=0, gauss_num=0, weight_decay=0., learning_rate=1e-3, epochs=100, batch_size=128):
    pre_X_train, y_train = data_reduction(pre_X_train, y_train, data_reduc=data_reduc, signal_vs_bg=signal_vs_bg)
    X_val, y_val = data_reduction(X_val, y_val, data_reduc=1.0, signal_vs_bg=signal_vs_bg)
    # y_train is now 190k*1 matrix 
    # normalize datasets here because raw X_train is needed in the evaluation 
    X_train, X_val = data_process.normalize_datasets(pre_X_train, X_val, pre_X_train)
    print("X_train and X_val are normalized.")

    # Class weight
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train[:, 0]), y=y_train[:, 0])
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    print("Class Weight: " + str(class_weights))
    
    if hingeloss:
        loss_fn = torch.nn.HingeEmbeddingLoss()
        y_train[y_train==0]=-1
        y_val[y_val==0]=-1
    else:
        loss_fn = define_model.BCELoss_class_weighted(class_weights)

    # define X, y
    if phi:
        X_train = torch.from_numpy(X_train[:, :5]).float().to(device)
        X_val = torch.from_numpy(X_val[:, :5]).float().to(device)
    elif uniform_num>0:
        X_train = torch.from_numpy(X_train[:, :4 + uniform_num]).float().to(device)
        X_val = torch.from_numpy(X_val[:, :4 + uniform_num]).float().to(device)
    else:
        X_train = torch.from_numpy(X_train[:, :4 + gauss_num]).float().to(device)
        X_val = torch.from_numpy(X_val[:, :4 + gauss_num]).float().to(device)

    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)

    # count number of inputs
    input_number = len(X_train[0])
    print("Input Number is " + str(input_number) + " nodes.")

    # prepare Dataset
    train_data = data_process.MakeASet(X_train, y_train)
    val_data = data_process.MakeASet(X_val, y_val)

    # make DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=len(val_data), shuffle=True)

    for model_num in model_nums:
        print(f"Model {model_num+1}\n-------------------------------")
        # redefining model everytime we start training new model to clear out the old graph
        if weight_seed:
            torch.manual_seed(19960609*model_num)

        if dropout:
            model = define_model.NeuralNetworkDropout(input_number, l1, l2, l3).to(device)
        elif morelayers:
            model = define_model.NeuralNetworkMoreLayers(input_number, nodes=l1).to(device)
        else:    
            model = define_model.NeuralNetwork(input_number, l1, l2, l3).to(device)
       
        if use_SGD:
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            print("use SGD optimizer")
        else:
            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            print("use Adam optimizer")
        
        train_loss_coll = []
        val_loss_coll = []

        # these change with data reduction
        size_train = len(train_loader.dataset) 
        n_batch = math.ceil(size_train/batch_size)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            model.train()
            train_loss, val_loss, train_loss_mean = 0, 0, 0

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
                
                # train_correct = (pred.round() == y).sum().item()/batch_size

            print(f"Training loss: {train_loss_mean/n_batch:>7f}")

            train_loss_coll.append(train_loss_mean/n_batch)

            # validation
            model.eval()
            # val_loss_mean = 0
            with torch.no_grad():
                for batch, (X, y) in enumerate(val_loader):
                    val_pred = model(X)
                    val_loss = loss_fn(val_pred, y)
                    # val_loss_mean += val_loss.mean().item()
                    # val_correct = (val_pred.round() == y).sum().item()
                    # val_correct /= len(X)

            # val_correct = val_correct/math.ceil(len(X_val)/batch_size)
            # val_loss_mean = val_loss_mean/math.ceil(len(X_val)/batch_size)
            print(f"Validation loss: {val_loss:>8f} \n")

            val_loss_coll.append(val_loss.item())
            # val_corr_coll.append(val_correct)

            # save model each epoch
            np.save(os.path.join('saved_model/' + save_dir, "train_losses_model_"+str(model_num+1)+".npy"), train_loss_coll)
            np.save(os.path.join('saved_model/'+ save_dir, "val_losses_model_"+str(model_num+1)+".npy"), val_loss_coll)
            torch.save(model.state_dict(), os.path.join('saved_model/' + save_dir, 'saved_training_model'+str(model_num+1)+'_epoch'+str(t+1)+'.pt'))

        print("Model "+str(model_num+1)+" training is Done!")

        # Loss values plot
        plt.plot(range(epochs), train_loss_coll, 'm.-', label = "average train loss")
        plt.plot(range(epochs), val_loss_coll, 'c.-', label = "val loss")
        plt.xlabel('epoch')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.legend(loc="upper left")
        plt.savefig('all_plots/' + save_dir + 'train_loss_value_model'+str(model_num+1)+'.png')
        plt.close()
        print("Finish plotting loss functions.")