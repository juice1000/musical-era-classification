# Year prediction using Keras neural nets
# This is the baseline file we used to run our experiments on condor.
# Every experiment on condor uses most of this code, with a few small modifications catered to that particular experiment. 

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from sklearn import preprocessing
from keras import regularizers
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import KFold


# Defining the sequential model
model = Sequential()

# Our examples of 90 features, so input_dim = 90
model.add(Dense(units=100, activation='relu', input_dim=90))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(units=100, activation='relu'))
#model.add(Flatten())

# Output is 0-2011, after conversion to categorical vars
model.add(Dense(units=9, activation='softmax'))

# Tune the optimizer
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
labels = []
examples = []
print ("GETTING DATASET")

# Replace filename with the path to the CSV where you have the year predictions data saved.
filename = "data_library/YearPredictionMSD.txt"
with open(filename, 'r') as f:
    for line in f:
        content = line.split(",")
        
        labels.append(content[0])

        content.pop(0)

        # If we wanted pure lists, and convert from string to float
        #content = [float(elem) for elem in content]
        #content = map(float, content)

        # If we want a list of numpy arrays, not necessary
        #npa = np.asarray(content, dtype=np.float64)

        examples.append(content)

print ("SPLITTING TRAINING AND TEST SETS")
# intilize a null list
unique = []
# traverse for all elements


for x in range(len(labels)):
    # check if exists in unique_list or not
    if labels[x] not in unique:
        unique.append(labels[x])

#for x in range(len(unique)):
#    print(unique[x])

#print("\n",len(unique))

# Turning lists into numpy arrays
total_array = np.array(examples)

total_labels = np.array(labels)
training_examples = total_array[:5000]
training_labels = total_labels[:5000]

labels = np.array(training_labels)
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


# Scale the features so they have 0 mean
total_scaled = preprocessing.scale(training_examples)

y_train = keras.utils.to_categorical(training_labels, num_classes=9)


kf = KFold(n_splits = 5)



# Train the model!
history = model.fit(training_examples, y_train, validation_split=0.33, epochs=20, batch_size=32)


# Loss and metrics
loss_and_metrics = model.evaluate(training_examples, y_train, batch_size=32)

print (loss_and_metrics)



print ("Creating Plots!")
print (history.history.keys())

#accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.savefig("model_acc.png")

#loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.savefig("model_loss.png")