# Year prediction using Keras neural nets
# This is the baseline file we used to run our experiments on condor.
# Every experiment on condor uses most of this code, with a few small modifications catered to that particular experiment.

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from CODE.models.year_pred_keras import Sequential
from CODE.models.year_pred_keras import Dense
from CODE.models.year_pred_keras import Dropout
from CODE.models.year_pred_keras import maxnorm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import os
import time

opt = year_pred_keras.optimizers.Adam(lr=0.001)

# Auxillary Function for getting model name througout validation
def get_model_name():
    return 'saved_models/model_' + '.h5'


def CNN_model():
    # Defining the sequential model
    model = Sequential()

    # Our examples of 90 features, so input_dim = 90
    model.add(Dense(units=200, activation='relu', input_dim=89))
    model.add(Dropout(0.1))
    model.add(Dense(units=300, activation='relu', kernel_constraint=maxnorm(1)))
    model.add(Dropout(0.1))
    model.add(Dense(units=300, activation='relu', kernel_constraint=maxnorm(1)))
    model.add(Dropout(0.1))
    model.add(Dense(units=300, activation='relu', kernel_constraint=maxnorm(1)))
    model.add(Dropout(0.1))
    model.add(Dense(units=300, activation='relu', kernel_constraint=maxnorm(1)))
    # model.add(Flatten())

    # Output is 0-8 for each decade
    model.add(Dense(units=9, activation='softmax'))

    # Tune the optimizer
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def CNN_TEST_model():
    # Defining the sequential model
    model = Sequential()

    # Our examples of 90 features, so input_dim = 90
    model.add(Dense(units=200, activation='relu', input_dim=89))
    model.add(Dense(units=300, activation='relu'))
    model.add(Dense(units=300, activation='relu'))
    model.add(Dense(units=300, activation='relu'))
    model.add(Dense(units=300, activation='relu'))
    # model.add(Flatten())

    # Output is 0-8 for each decade
    model.add(Dense(units=9, activation='softmax'))

    # Tune the optimizer
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

def main(filename, test):
    labels = []
    examples = []
    print("GETTING DATASET")
    with open(filename, 'r') as f:
        for line in f:
            content = line.split(",")

            labels.append(content[0])
            content.pop(0)

            # If we wanted pure lists, and convert from string to float
            # content = [float(elem) for elem in content]
            # content = map(float, content)

            # If we want a list of numpy arrays, not necessary
            # npa = np.asarray(content, dtype=np.float64)

            examples.append(content)

    print("SPLITTING TRAINING AND VALIDATION SETS")
    # intilize a null list
    unique = []
    # traverse for all elements


    #for x in range(len(labels)):
        # check if exists in unique_list or not
    #   if labels[x] not in unique:
    #       unique.append(labels[x])
    # for x in range(len(unique)):
    #    print(unique[x])
    # print("\n",len(unique))


    # Turning lists into numpy arrays
    training_examples = np.array(examples)
    training_labels = np.array(labels)

    #print(training_labels)
    #print(training_examples)

    training_examples = preprocessing.scale(training_examples)


    for x in range(len(training_labels)):
        if int(training_labels[x]) < 1930:
            training_labels[x] = 0
        elif int(training_labels[x]) < 1940 and int(training_labels[x]) >= 1930:
            training_labels[x] = 1
        elif int(training_labels[x]) < 1950 and int(training_labels[x]) >= 1940:
            training_labels[x] = 2
        elif int(training_labels[x]) < 1960 and int(training_labels[x]) >= 1950:
            training_labels[x] = 3
        elif int(training_labels[x]) < 1970 and int(training_labels[x]) >= 1960:
            training_labels[x] = 4
        elif int(training_labels[x]) < 1980 and int(training_labels[x]) >= 1970:
            training_labels[x] = 5
        elif int(training_labels[x]) < 1990 and int(training_labels[x]) >= 1930:
            training_labels[x] = 6
        elif int(training_labels[x]) < 2000 and int(training_labels[x]) >= 1930:
            training_labels[x] = 7
        else:
            training_labels[x] = 8


    cvscores = []
    cvloss = []

    save_dir = '../../saved_models'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # split into input (X) and output (Y) variables
    X = training_examples
    Y = training_labels

    # CREATE NEW MODEL
    model = CNN_model()
    checkpoint = year_pred_keras.callbacks.ModelCheckpoint(get_model_name(), monitor='acc', save_best_only=True, verbose=0, mode='max')
    callbacks = [checkpoint]

    if test is False:
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        fold_var = 1


        for train, test in kfold.split(X, Y):

            # Convert to categorical in order to produce proper output
            print(Y[train])
            y_train = year_pred_keras.utils.to_categorical(Y[train], num_classes=9)
            print(y_train)
            y_test = year_pred_keras.utils.to_categorical(Y[test], num_classes=9)

            # COMPILE MODEL
            model.fit(X[train], y_train, epochs=10, callbacks=callbacks, verbose=0)
            # evaluate model
            scores = model.evaluate(X[test], y_test, verbose=0)
            print("%s: %.2f%%, loss:" % (model.metrics_names[1], scores[1] * 100), scores[0])
            cvscores.append(scores[1] * 100)
            cvloss.append(scores[0])

            model.load_weights(save_dir+"/model_" + ".h5")
            fold_var += 1

        print("\n%%%%%% PERFORMANCE %%%%%%")
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

        time.sleep(1)
        print("\n%%%%%% CREATE PLOTS %%%%%%")

        plt.figure(1)
        plt.plot(cvscores)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.title("Training -- Accuracy")

        plt.figure(2)
        plt.plot(cvloss)
        plt.title("Loss")
        plt.ylabel("Loss")
        plt.title("Training -- Loss")


    else:
        model = CNN_TEST_model()
        model.load_weights(save_dir + "/model_" + ".h5")
        y_test = year_pred_keras.utils.to_categorical(Y, num_classes=9)

        predictions = model.predict(X)
        predictions = np.array(predictions)
        predictions = np.around(predictions)

        print(predictions)
        total = len(y_test)
        count = 0


        for i in range(len(predictions)):
            #print("X=%s, Predicted=%s" % (predictions[i], y_test[i]))
            if np.all(predictions[i] == y_test[i]):
                count += 1

        accuracy = (count/total) * 100
        print("\n\nMODEL ACCURACY - TEST: ", round(accuracy, 2), "%")


        scores = model.evaluate(X, y_test, verbose=0)
        print("EVALUATION - %s: %.2f%%, loss:" % (model.metrics_names[1], scores[1] * 100), scores[0])
        cvscores.append(scores[1] * 100)
        cvloss.append(scores[0])


if __name__ == "__main__":

    print("\n INITIALIZE TRAINING")
    main(filename="data_library/MSD_unbiased.csv", test=False)

    plt.show()

    print("\n INITIALIZE TESTING")
    main(filename="data_library/MSDB_5000_testset.csv", test=True)

