
import multiprocessing as mp
import pandas as pd

df = pd.read_csv('../data_library/preprocessing_data/url_data_CNN.csv')

columns = ["Year", "ID", "Artist", "Title", "URL"]
df.columns = columns
df["Specs"] = df["Artist"] + " -- " + df["Title"] + ".png"
training_labels = df



print(training_labels)