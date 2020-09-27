import librosa
import os, os.path
import pandas as pd
import statistics
import numpy as np

def normalize(item):

    item_normalized = []    
    for i in range(len(item)):
        item_normalized.append(np.mean(item[i]) / np.var(item[i]))

    return item_normalized


def create_col_names(item):
    
    names = []
    for i in range(len(item[0])):
        names.append("mfcc_" + str(i+1))

    return names


DIR = 'music_library'
# simple version for working with CWD
files = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
print(len(files))
# Append features to dataframe
df = pd.read_csv("data_library/list_downloaded_previews.csv")
df2 = df.values

chroma_stft_arr = []
spec_cent_arr = []
spec_bw_arr = []
rolloff_arr = []
zcr_arr = []
mfcc_arr = []

for i in range(len(df2)):

    filename = df2[i][2] + " -- " + df2[i][3] + ".wav"
    if os.path.isfile(os.path.join(DIR, filename)):

        audio_path = "./" + DIR + "/" + filename

        print(filename)
        y , sr = librosa.load(os.fspath(audio_path))

        # Get all librosa features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_arr.append(normalize(chroma_stft))

        #rmse = librosa.feature.rmse(y=y)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_arr.append(normalize(spec_cent))

        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bw_arr.append(normalize(spec_bw))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_arr.append(normalize(rolloff))

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_arr.append(normalize(zcr))

        # This one has 20 features to put in different columns
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_arr.append(normalize(mfcc))


# Create Dataframes from audio features
mfcc_names = create_col_names(mfcc_arr)
mfcc = pd.DataFrame(mfcc_arr, columns=mfcc_names)

chroma_names = create_col_names(chroma_stft_arr)
chroma_stft = pd.DataFrame(chroma_stft_arr, columns=chroma_names)

spec_cent = pd.DataFrame(np.array(spec_cent_arr), columns=["spec_cent"])
spec_bw = pd.DataFrame(np.array(spec_bw_arr), columns=["spec_bw"])
rolloff = pd.DataFrame(np.array(rolloff_arr), columns=["rolloff"])
zcr = pd.DataFrame(np.array(zcr_arr), columns=["zcr"])

# Concatenate Dataframes
df = pd.concat([df, mfcc, chroma_stft, spec_cent, spec_bw, rolloff, zcr], axis=1)
print(df.head())
df.to_csv("data_library/librosa_audiofeatures.csv")


