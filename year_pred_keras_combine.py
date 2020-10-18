# Year prediction using Keras neural nets
# This is the baseline file we used to run our experiments on condor.
# Every experiment on condor uses most of this code, with a few small modifications catered to that particular experiment.

import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

import numpy as np
from sklearn import preprocessing

from sklearn.model_selection import StratifiedKFold
import os
import datetime
import pandas as pd

tf.debugging.set_log_device_placement(True)

save_dir = 'saved_models'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Auxillary Function for getting model name througout validation
def get_model_name():
    return save_dir + '/model.h5'


def CNN_model(width, length, features, test):
    # Defining the sequential model
    model = Sequential()

    if test is False:
        model.add(Dense(units=width, activation='relu', input_dim=36))
        model.add(BatchNormalization())
        for i in range(length):
                model.add(Dense(units=width, activation='relu'))
                model.add(BatchNormalization())
    else:
        model.add(Dense(units=width, activation='relu', input_dim=36))
        model.add(BatchNormalization())
        for i in range(length):
                model.add(Dense(units=width, activation='relu'))
                model.add(BatchNormalization())

    # Output is 0-8 for each decade
    model.add(Dense(units=features, activation='softmax'))

    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model


def data_preprocessing(filename):
    print("GETTING DATASET")

    data = pd.read_csv(filename)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns="Unnamed: 0")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    print(data.head())
    print("SPLITTING TRAINING AND VALIDATION SETS")

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
    with tf.device("/cpu:0"):
        # We test for 100 - 400 nodes
        # With 2-5 hidden layers
        start = datetime.datetime.now()

        df = pd.DataFrame(index=["1 Hidden", "2 Hidden", "3 Hidden"], columns=["100", "200", "300", "400", "500"])


        for i in range(1, 6):
            width = i * 100

            for length in range(2, 5):

                X, Y, features = data_preprocessing(training_data)
                # CREATE NEW MODEL
                model = CNN_model(width=width, length=length, features=features, test=False)
                es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
                mc = ModelCheckpoint(get_model_name(), monitor='accuracy', mode='max', verbose=0, save_best_only=True)
                callbacks = [es, mc]

                # fix random seed for reproducibility
                seed = 7
                np.random.seed(seed)
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
                fold_var = 1


                for train, test in kfold.split(X, Y):

                    cvscores = []
                    cvloss = []

                    # Convert to categorical in order to produce proper output
                    y_train = keras.utils.to_categorical(Y[train], num_classes=features)
                    y_test = keras.utils.to_categorical(Y[test], num_classes=features)

                    with tf.device("/gpu:0"):
                        # COMPILE MODEL
                        model.fit(X[train], y_train, epochs=20, callbacks=callbacks, verbose=0)

                    # evaluate model
                    scores = model.evaluate(X[test], y_test, verbose=0)
                    print("%s: %.2f%%, loss:" % (model.metrics_names[1], scores[1] * 100), scores[0])
                    cvscores.append(scores[1] * 100)
                    cvloss.append(scores[0])

                    #model.save_weights(get_model_name())
                    fold_var += 1

                print("\n%%%%%% PERFORMANCE %%%%%%")
                print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

                print("\n%%%%%% CREATE PLOTS %%%%%%")

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
                model = CNN_model(width=width, length=length, features=features, test=True)
                model.load_weights(get_model_name())
                y_test = keras.utils.to_categorical(Y, num_classes=features)

                #predictions = model.predict(X)
                #predictions = np.array(predictions)
                #le = preprocessing.LabelBinarizer()
                #le.fit([0, 1, 2, 3, 4, 5, 6, 7, 8])
                #predictions = le.inverse_transform(predictions)
                #print(predictions)

                #m = tf.keras.metrics.Accuracy()
                #m.update_state(Y, predictions)
                #print("\n\nMODEL ACCURACY - TEST: ", round(m.result().numpy(), 4) * 100, "%" )


                scores = model.evaluate(X, y_test, verbose=0)
                print("EVALUATION - %s: %.2f%%, loss:" % (model.metrics_names[1], scores[1] * 100), scores[0])
                cvscores.append(scores[1] * 100)
                cvloss.append(scores[0])

                df.loc["{} Hidden".format(length-1), str(width)] = scores

        df.to_csv("NN_eval/NN_eval_librosa_audiofeatures.csv")
        #plt.show()


if __name__ == "__main__":

    print("\n INITIALIZE TRAINING")

    main(training_data="data_library/librosa_audiofeatures_train5feat.csv",
         test_data="data_library/librosa_audiofeatures_test5feat.csv")


