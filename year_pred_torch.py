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
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(36, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 5),
            nn.LogSoftmax(),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)



def data_preprocessing(filename):
    print("GETTING DATASET")

    data = pd.read_csv(filename)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns="Unnamed: 0")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    #print(data.head())

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

    return X, Y, features


def main(training_data, test_data):

        # We test for 100 - 400 nodes
        # With 2-5 hidden layers
        start = datetime.datetime.now()

        df = pd.DataFrame(index=["1 Hidden", "2 Hidden", "3 Hidden"], columns=["100", "200", "300", "400", "500"])


        for i in range(1, 6):
            width = i * 100

            for length in range(2, 5):

                X, Y, features = data_preprocessing(training_data)

                #kf = StratifiedKFold(n_splits=8, random_state=3829, shuffle=True)
                # CREATE NEW MODEL
                net = Net()

                optimizer = torch.optim.Adam(
                    net.parameters(), betas=(0.9, 0.999), lr=1e-4, weight_decay=0)
                criterion = nn.NLLLoss()

                #model = CNN_model(width=width, length=length, features=features, test=False)
                #es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
                #mc = ModelCheckpoint(get_model_name(), monitor='accuracy', mode='max', verbose=0, save_best_only=True)
                #callbacks = [es, mc]

                # fix random seed for reproducibility
                seed = 7
                np.random.seed(seed)
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
                fold_var = 1


                for train, test in kfold.split(X, Y):

                    cvscores = []
                    cvloss = []


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
                        net.train()

                        inputs, labels = data
                        labels = labels.to(device, dtype=torch.int64)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        with torch.no_grad():
                            net.eval()

                            for i, data in enumerate(val_loader, 0):
                                inputs, labels = data
                                labels = labels.to(device, dtype=torch.int64)

                                outputs_val = net(inputs)
                                val_loss = criterion(outputs_val, labels)
                                #loss.backward()
                                #optimizer.step()
                                cvloss.append(val_loss.item())

                    print("LOSS: ", np.mean(cvloss))

                    #model.save_weights(get_model_name())
                    fold_var += 1



                #print("\n%%%%%% CREATE PLOTS %%%%%%")

                #plt.figure(1)
                #plt.plot(cvscores)
                #plt.title('model accuracy')
                #plt.ylabel('accuracy')
                #plt.title("Training -- Accuracy")

                #plt.figure(2)
                #plt.plot(cvloss)
                #plt.title("Loss")
                #plt.ylabel("Loss")
                #plt.title("Training -- Loss")

                end = datetime.datetime.now()
                print("Elapsed Time: ", ((end-start).total_seconds())/60)


    #######################    TESTING     ########################

                X, Y, features = data_preprocessing(test_data)
                test_loader = get_dataloader(
                    X, Y,
                    batch_size=200, shuffle=False
                )
                predictions, val_losses = [], []
                #model.load_weights(get_model_name())
                #y_test = keras.utils.to_categorical(Y, num_classes=features)

                count = 0
                labelsize = 0

                with torch.no_grad():
                    net.eval()
                    for i, data in enumerate(test_loader, 0):
                        inputs, labels = data
                        labels_ = labels.to(device, dtype=torch.int64)

                        predictions = net(inputs)
                        test_loss = criterion(predictions, labels_)

                        _, predicted = torch.max(predictions.data, 1)
                        labelsize += labels.size(0)
                        count += (predicted == labels).sum().item()


                    accuracy = 100 * count / labelsize
                    print('TEST ACCURACY: ', (accuracy))
                    print('TEST LOSS: %d %%' % (test_loss))

                #predictions = model.predict(X)
                #predictions = np.array(predictions)
                #le = preprocessing.LabelBinarizer()
                #le.fit([0, 1, 2, 3, 4, 5, 6, 7, 8])
                #predictions = le.inverse_transform(predictions)
                #print(predictions)

                #m = tf.keras.metrics.Accuracy()
                #m.update_state(Y, predictions)
                #print("\n\nMODEL ACCURACY - TEST: ", round(m.result().numpy(), 4) * 100, "%" )


                #scores = model.evaluate(X, y_test, verbose=0)
                #print("EVALUATION - %s: %.2f%%, loss:" % (model.metrics_names[1], scores[1] * 100), scores[0])
                #cvscores.append(scores[1] * 100)
                #cvloss.append(scores[0])

                df.loc["{} Hidden".format(length-1), str(width)] = accuracy

        df.to_csv("NN_eval/NN_eval_librosa_audiofeatures.csv")
        #plt.show()


if __name__ == "__main__":

    print("\n INITIALIZE TRAINING")

    main(training_data="data_library/librosa_audiofeatures_train5feat.csv",
         test_data="data_library/librosa_audiofeatures_test5feat.csv")


