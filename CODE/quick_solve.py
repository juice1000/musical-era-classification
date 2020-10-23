import pandas as pd
import numpy as np

training_data = "data_library/librosa_audiofeatures_test5feat.csv"
data = pd.read_csv(training_data)
if "Unnamed: 0" in data.columns:
    data = data.drop(columns="Unnamed: 0")
data = data.replace([np.inf, -np.inf], np.nan).dropna()

data.to_csv("data_library/librosa_audiofeatures_test5feat.csv")