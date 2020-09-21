import librosa
import os, os.path
import pandas as pd


DIR = 'music_library'
# simple version for working with CWD
files = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]

# Append features to dataframe
df = pd.read_csv("data_library/list_downloaded_previews.csv")


for i in range(len(df)):

    filename = df[i][2] + " -- " + df[i][3] + ".mp3"
    print("FILENAME: ", filename)
    if os.path.isfile(os.path.join(DIR, filename)):
        print(df[i][2] + " -- " + df[i][3] + ".mp3")

        audio_path = DIR + "/" + df[i][2] + " -- " + df[i][3] + ".mp3"
        y , sr = librosa.load(audio_path)
        #print(type(x), type(sr))

        # Get all librosa features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        # This one has 20 features to put in different columns
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        print(chroma_stft)
        print(rmse)
        print(spec_cent)
        print(spec_bw)
        print(rolloff)
        print(zcr)
        print(mfcc)


