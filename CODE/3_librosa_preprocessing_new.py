import librosa
import os.path
import pandas as pd
import numpy as np

def normalize(item):

    item_normalized = []    
    for i in range(len(item)):
        item_normalized.append(np.mean(item[i]) / np.var(item[i]))

    return item_normalized


def create_col_names(num, name):
    
    names = []
    for i in range(num):
        names.append(name + str(i+1))

    return names


def assign_colnames(arr, item):
    for i in range(len(item)):
        arr.append(item[i])

    return arr


DIR = './music_library'
# simple version for working with CWD
files = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
print(len(files))
# Append features to dataframe
df = pd.read_csv("./data_library/preprocessing_data/url_data_20s.csv")
headers = ["Year", "ID", "Artist", "Title", "URL"]
df.columns = headers

i = 0
batch = 10000

mfcc_names = create_col_names(20, "mfcc_")
chroma_names = create_col_names(12, "chroma_stft_")
features = ["spec_cent", "spec_bw", "rolloff", "zcr"]

columns = []

columns = assign_colnames(columns, headers)
columns = assign_colnames(columns, chroma_names)
columns = assign_colnames(columns, features)
columns = assign_colnames(columns, mfcc_names)

for k in range(224):

    min = i
    max = i + batch
    df_part = df[min:]
    df2 = df_part.values

    df_part = df_part.reindex(columns=columns)
    df3 = df_part.values

    for j in range(224):

        i = j + k * batch

        filename = df2[j][2] + " -- " + df2[j][3] + ".wav"
        if os.path.isfile(os.path.join(DIR, filename)):

            audio_path = "./" + DIR + "/" + filename

            print(i, filename)
            y, sr = librosa.load(os.fspath(audio_path))

            # Get all librosa features
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_stft = normalize(chroma_stft)

            #rmse = librosa.feature.rmse(y=y)

            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_cent = normalize(spec_cent)

            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spec_bw = normalize(spec_bw)

            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff = normalize(rolloff)

            zcr = librosa.feature.zero_crossing_rate(y)
            zcr = normalize(zcr)

            # This one has 20 features to put in different columns
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc = normalize(mfcc)

            df3[j] = np.hstack([df2[j], chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc])



    # Create Dataframe from audio features
    df3 = pd.DataFrame(df3, columns=columns)
    df_name = "data_library/librosa_audiofeatures_{}.csv".format(i)
    df3.to_csv(df_name)

