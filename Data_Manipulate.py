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
filename = "YearPredictionMSD.txt"
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

training_examples = examples
training_labels = labels


twen = 0
third = 0
four = 0
fif = 0
six = 0
seven = 0
eigt = 0
ninet = 0
thou = 0

for x in range(len(training_labels)):
    if int(training_labels[x]) < 1930:
        twen += 1
    elif int(training_labels[x]) < 1940 and int(training_labels[x]) >= 1930:
        third += 1
    elif int(training_labels[x]) < 1950 and int(training_labels[x]) >= 1940:
        four += 1
    elif int(training_labels[x]) < 1960 and int(training_labels[x]) >= 1950:
        fif += 1
    elif int(training_labels[x]) < 1970 and int(training_labels[x]) >= 1960:
        six += 1
    elif int(training_labels[x]) < 1980 and int(training_labels[x]) >= 1970:
        seven += 1
    elif int(training_labels[x]) < 1990 and int(training_labels[x]) >= 1930:
        eigt += 1
    elif int(training_labels[x]) < 2000 and int(training_labels[x]) >= 1930:
        ninet += 1
    else:
        thou += 1


print(twen, third, four, fif, six, seven, eigt, ninet, thou)

# Create new subdata
training_examples = pd.DataFrame(training_examples)
training_examples = training_examples.drop(axis=1, columns=90)
print(training_examples.head())
training_examples.to_csv('data_library/MSDB_YearPred.csv', index=False, header=False)

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



#labels = np.array(training_labels)

# Scale the features so they have 0 mean
#total_scaled = preprocessing.scale(training_examples)



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

