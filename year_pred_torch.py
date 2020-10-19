# Year prediction using Keras neural nets
# This is the baseline file we used to run our experiments on condor.
# Every experiment on condor uses most of this code, with a few small modifications catered to that particular experiment.

import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from sklearn import preprocessing

from sklearn.model_selection import StratifiedKFold
import os
import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp


save_dir = 'saved_models'
device = "cpu"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Auxillary Function for getting model name througout validation
def get_model_name():
    return save_dir + '/model.h5'


def get_dataset(x, y):
    return TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float()
    )


def get_dataloader(x: np.array, y: np.array, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = get_dataset(x, y)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


class Net(nn.Module):
    def __init__(self, length, width):
        super().__init__()

        self.width = width
        self.length = length

        modules = []
        modules.append(nn.Linear(36, self.width))
        modules.append(nn.ReLU())
        for i in range(self.length):
            modules.append(nn.Linear(self.width, self.width))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(self.width, 5))
        modules.append(nn.Softmax(dim=0))

        self.model = nn.Sequential(*modules)

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


def data_preprocessing(filename):
    print("GETTING DATASET")
    data = filename
    dataset = data.values

    training_examples = dataset[:,1:]
    training_labels = dataset[:, 0]

    training_examples = preprocessing.scale(training_examples)

    for x in range(len(training_labels)):
        if int(training_labels[x]) < 1930:
            training_labels[x] = 0
        elif int(training_labels[x]) < 1940 and int(training_labels[x]) >= 1930:
            training_labels[x] = 1
        elif int(training_labels[x]) < 1950 and int(training_labels[x]) >= 1940:
            training_labels[x] = 2
        elif int(training_labels[x]) < 1960 and int(training_labels[x]) >= 1950:
            training_labels[x] = 0
        elif int(training_labels[x]) < 1970 and int(training_labels[x]) >= 1960:
            training_labels[x] = 1
        elif int(training_labels[x]) < 1980 and int(training_labels[x]) >= 1970:
            training_labels[x] = 2
        elif int(training_labels[x]) < 1990 and int(training_labels[x]) >= 1930:
            training_labels[x] = 3
        elif int(training_labels[x]) < 2000 and int(training_labels[x]) >= 1930:
            training_labels[x] = 4
        else:
            training_labels[x] = 8

    features = len(np.unique(training_labels))
    print("\nNUMBER OF FEATURES: ", features)

    # split into input (X) and output (Y) variables
    X = training_examples
    Y = training_labels

    return X, Y


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, name, shards, criterion, width, length):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep = global_ep
        self.gnet, self.optimizer = gnet, opt
        self.width = width
        self.length = length
        self.lnet = Net(self.length, self.width)  # local network
        self.shards = shards
        self.criterion = criterion

    def run(self):

        X, Y = data_preprocessing(self.shards)

        while self.g_ep.value <= len(Y):

            start = datetime.datetime.now()

            # fix random seed for reproducibility
            seed = 7
            np.random.seed(seed)
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            fold_var = 1

            cvloss = []
            for train, test in kfold.split(X, Y):



                ##################################################

                train_loader = get_dataloader(
                    X[train], Y[train],
                    batch_size=2000, shuffle=True
                )
                val_loader = get_dataloader(
                    X[test], Y[test],
                    batch_size=100, shuffle=False
                )

                for i, data in enumerate(train_loader, 0):
                    self.lnet.train()

                    inputs, labels = data
                    labels = labels.to(device, dtype=torch.int64)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.lnet(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()



                    for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
                        gp._grad = lp.grad
                    self.optimizer.step()
                    self.lnet.load_state_dict(self.gnet.state_dict())

                    with self.g_ep.get_lock():
                        self.g_ep.value += 1

                    with torch.no_grad():
                        self.lnet.eval()

                        for i, data in enumerate(val_loader, 0):
                            inputs, labels = data
                            labels = labels.to(device, dtype=torch.int64)

                            outputs_val = self.lnet(inputs)
                            val_loss = self.criterion(outputs_val, labels)
                            cvloss.append(val_loss.item())

                            with self.g_ep.get_lock():
                                self.g_ep.value += 1

                fold_var += 1
            print("LOSS: ", np.mean(cvloss))


                # model.save_weights(get_model_name())

            end = datetime.datetime.now()
            print("\nWorker: ", self.name, " | Elapsed Time: ", ((end - start).total_seconds()) / 60)




if __name__ == "__main__":

    print("\n INITIALIZE TRAINING")

    # We test for 100 - 400 nodes
    # With 2-5 hidden layers
    df = pd.DataFrame(index=["1 Hidden", "2 Hidden", "3 Hidden"], columns=["100", "200", "300", "400", "500"])

    training_data = "data_library/librosa_audiofeatures_train5feat.csv"
    data = pd.read_csv(training_data)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns="Unnamed: 0")
    #data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # create as many processes as there are CPUs on your machine
    num_processes = mp.cpu_count()
    # calculate the chunk size as an integer
    shard_size = int(data.shape[0] / num_processes)
    # this solution was reworked from the above link.
    # will work even if the length of the dataframe is not evenly divisible by num_processes
    shards = [data.iloc[data.index[i:i + shard_size]] for i in range(0, data.shape[0], shard_size)]

    for i in range(1, 6):
        width = i * 100

        for length in range(2, 5):
            gnet = Net(length, width)
            print(gnet)
            criterion = nn.CrossEntropyLoss()

            gnet.share_memory()  # share the global parameters in multiprocessing
            opt = SharedAdam(gnet.parameters(), lr=0.005, betas=(0.92, 0.999))  # global optimizer
            global_ep, global_ep_r = mp.Value('i', 0), mp.Value('d', 0.)

            workers = [Worker(gnet, opt, global_ep, i, shards[i], criterion, width, length)
            for i in range(mp.cpu_count())]
            [w.start() for w in workers]

            [w.join() for w in workers]

            #######################    TESTING     ########################

            test_data = pd.read_csv("data_library/librosa_audiofeatures_test5feat.csv")
            if "Unnamed: 0" in test_data.columns:
                test_data = test_data.drop(columns="Unnamed: 0")

            X_test, Y_test = data_preprocessing(test_data)

            test_loader = get_dataloader(
                X_test, Y_test,
                batch_size=200, shuffle=False
            )

            count = 0
            labelsize = 0
            test_loss = []
            scores = []

            with torch.no_grad():
                gnet.eval()
                for i, data in enumerate(test_loader, 0):
                    inputs, labels = data
                    labels_ = labels.to(device, dtype=torch.int64)

                    predictions = gnet(inputs)
                    loss = criterion(predictions, labels_)
                    test_loss.append(loss.item())

                    _, predicted = torch.max(predictions.data, 1)
                    labelsize += labels.size(0)
                    count += (predicted == labels).sum().item()

                accuracy = round((100 * count / labelsize), 2)
                final_loss = round((np.mean(test_loss)),2)
                print('\n\nTEST ACCURACY: ', accuracy, "%")
                print('TEST LOSS: ', final_loss)

                scores.append(accuracy)
                scores.append(final_loss)

            df.loc["{} Hidden".format(length - 1), str(width)] = scores

    df.to_csv("NN_eval/NN_eval_librosa_audiofeatures_torch.csv")



