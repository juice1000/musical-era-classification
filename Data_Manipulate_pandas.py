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

column_names = ["Year"]

for i in range(1,90):
    column_names.append(str(i))


df = pd.read_csv("data_library/MSD_unbiased.csv", header=None)
df.columns = column_names

twen = df.loc[df['Year'] < 1930]
third = df.loc[(df['Year'] >= 1930) & (df['Year'] < 1940)]
four = df.loc[(df['Year'] >= 1940) & (df['Year'] < 1950)]
fif = df.loc[(df['Year'] >= 1950) & (df['Year'] < 1960)]
six = df.loc[(df['Year'] >= 1960) & (df['Year'] < 1970)]
seven = df.loc[(df['Year'] >= 1970) & (df['Year'] < 1980)]
eigt = df.loc[(df['Year'] >= 1980) & (df['Year'] < 1990)]
ninet = df.loc[(df['Year'] >= 1990) & (df['Year'] < 2000)]
thou = df.loc[df['Year'] >= 2000]

print(len(twen), len(third), len(four), len(fif), len(six), len(seven), len(eigt), len(ninet), len(thou))


four = four[0:300]
fif = fif[0:300]
six = six[0:3000]
seven = seven[0:300]
eigt = eigt[0:300]
ninet = ninet[0:300]
thou = thou[0:300]

print(len(twen), len(third), len(four), len(fif), len(six), len(seven), len(eigt), len(ninet), len(thou))


newDF = pd.concat([twen, third, four, fif, six, seven, eigt, ninet, thou], axis=0, ignore_index=True)
newDF = newDF.sample(frac=1)



# Create new subdata
#training_examples = pd.DataFrame(training_examples)
#training_examples = training_examples.drop(axis=1, columns=90)
#print(training_examples.head())
newDF.to_csv('data_library/MSD_unbiased_300.csv', index=False, header=False)


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

#
#

