# Year prediction using Keras neural nets
# This is the baseline file we used to run our experiments on condor.
# Every experiment on condor uses most of this code, with a few small modifications catered to that particular experiment.

import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from keras import regularizers
from keras.utils import np_utils, generic_utils
import os


# global va

# Auxillary Function for getting model name througout validation
def get_model_name(k):
    return 'saved_models/model_' + str(k) + '.h5'


def CNN_model():
    # Defining the sequential model
    model = Sequential()

    # Our examples of 90 features, so input_dim = 90
    model.add(Dense(units=50, activation='relu', input_dim=90))
    model.add(Dense(units=200, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Dense(units=200, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Dense(units=200, activation='relu'))
    #model.add(Dense(units=50, activation='relu'))
    # model.add(Flatten())

    # Output is 0-2011, after conversion to categorical vars
    model.add(Dense(units=2011, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


labels = []
examples = []
print("GETTING DATASET")

# Replace filename with the path to the CSV where you have the year predictions data saved.
filename = "data_library/YearPredictionMSD.txt"
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

print("SPLITTING TRAINING AND TEST SETS")
# Turning lists into numpy arrays
train_data = np.array(examples)

# Scale the features so they have 0 mean
total_scaled = preprocessing.scale(train_data)
# print(total_scaled)

# Numpy array of the labels
total_labels = np.array(labels)

# Split training and test:
# Increase or decrease these sizes affects run-time
training_examples = total_scaled[:200]
# training_examples = random.sample(total_array, 10)
training_labels = total_labels[:200]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
cvloss = []

save_dir = 'saved_models'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


fold_var = 1

y_train = keras.utils.to_categorical(training_labels, num_classes=2011)

# split into input (X) and output (Y) variables
X = training_examples
Y = training_labels


for train, test in kfold.split(X, Y):

    # Convert to categorical in order to produce proper output
    y_train = keras.utils.to_categorical(Y, num_classes=2011)
    # CREATE NEW MODEL
    model = CNN_model()

    checkpoint = keras.callbacks.ModelCheckpoint(get_model_name(fold_var),
                                                    monitor='accuracy', verbose=1,
                                                    save_best_only=True, mode='max')
    callbacks = [checkpoint]
    # COMPILE NEW MODEL
    model.fit(X[train], y_train[train], epochs=20, callbacks=callbacks, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], y_train[test], verbose=0)
    print("%s: %.2f%%, loss:" % (model.metrics_names[1], scores[1] * 100), scores[0])
    cvscores.append(scores[1] * 100)
    cvloss.append(scores[0])

    model.load_weights(save_dir+"/model_" + str(fold_var) + ".h5")
    fold_var+=1

# Train the model!
# history = model.fit(training_examples, y_train, validation_split=0.33, epochs=50, batch_size=32)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


print("Creating Plots!")
plt.figure(1)
plt.plot(cvscores)
plt.title('model accuracy')
plt.ylabel('accuracy')
# print (history.history.keys())

plt.figure(2)
plt.plot(cvloss)
plt.title("Loss")
plt.ylabel("Loss")

plt.show()
