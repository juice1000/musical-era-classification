import pandas as pd
import numpy as np
import scipy.sparse as sparse
from time import sleep
import os
from sklearn import preprocessing
import pandas as pd


labels = []
examples = []
print("GETTING DATASET")

# Replace filename with the path to the CSV where you have the year predictions data saved.
filename = "data_library/YearPredictionMSD.txt"
with open(filename, 'r') as f:
    for line in f:
        content = line.split(",")

        labels.append(content[0])
        #content.pop(0)

        # If we wanted pure lists, and convert from string to float
        # content = [float(elem) for elem in content]
        # content = map(float, content)

        # If we want a list of numpy arrays, not necessary
        # npa = np.asarray(content, dtype=np.float64)

        examples.append(content)

training_examples = examples[5000:7000]
training_labels = labels[5000:7000]

# Create new subdata
training_examples = pd.DataFrame(training_examples)
training_examples = training_examples.drop(axis=1, columns=90)
print(training_examples.head())
training_examples.to_csv('data_library/MSDB_5000_testset.csv', index=False, header=False)

# intilize a null list 
unique = [] 
# traverse for all elements 


for x in range(len(labels)):
    # check if exists in unique_list or not
    if labels[x] not in unique: 
        unique.append(labels[x])

#for x in range(len(unique)):
#    print(unique[x])
print("\n",len(unique))

# Turning lists into numpy arrays
total_array = np.array(examples)
total_labels = np.array(labels)



labels = np.array(training_labels)

# Scale the features so they have 0 mean
total_scaled = preprocessing.scale(training_examples)



#count = 0
#new_row_count = 0
#
#for i in range(len(examples)):
#    examples[i][0] = int(examples[i][0])
#
#for i in range(len(examples)):
#    if examples[i][0] > 1990:
#        new_row_count +=1
#
#new_examples = np.empty((new_row_count, len(examples[0])))
#
#for i in range(len(examples)):
#    if examples[i][0] > 1990:
#        for j in range(len(examples[0])):
#            new_examples[count][j] = examples[i][j]
#        count = count + 1
#
#
#df = pd.DataFrame(new_examples)
#df.to_csv("MSD1990.csv")
#
#

