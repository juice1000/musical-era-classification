
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
import  numpy as np
from matplotlib import figure
import gc
import pandas as pd
import os


def create_spectrogram(audiopath,filename):
    plt.interactive(False)
    clip, sample_rate = librosa.load(audiopath, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    image_path = '../melspectograms/400dpi/' + filename + '.png'
    plt.savefig(image_path, dpi=400, quality=95, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del image_path,filename,clip,sample_rate,fig,ax,S



DIR = '../music_library'
# simple version for working with CWD
files = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
print(len(files))
# Append features to dataframe
df = pd.read_csv("../data_library/preprocessing_data/url_data_CNN2.csv")


i = 0
batch = 300


for k in range(9):

    min = i
    max = i + batch
    df_part = df[min:]
    df2 = df_part.values

    #df3 = df_part.values

    for j in range(batch):
        i = j + k * batch

        filename = df2[j][2] + " -- " + df2[j][3] + ".wav"
        if os.path.isfile(os.path.join(DIR, filename)):
            print(i, filename)
            audio_path = "./" + DIR + "/" + filename
            filename = filename[:-4]
            create_spectrogram(audio_path, filename)

    gc.collect()

