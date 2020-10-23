import pandas as pd

def make_url(art_tracks):
    url = "https://api.spotify.com/v1/search?q="
    #print(art_tracks)
    for i in range(len(art_tracks)):
        url += art_tracks[i] + "%20"

    url += "&type=track"
    return url


df = pd.read_csv("../data_library/MSDB/MSDB_2000s.csv", header=None)
df2 = df[df.columns[2:]].apply(lambda x: ' '.join(x.astype(str)), axis=1)
data = df2.values
artists_tracks = []

for i in range(len(data)):
    artstring = data[i]
    arttracks = artstring.split()
    artists_tracks.append(arttracks)

#print(artists_tracks[1][2], "\n")
url = "https://api.spotify.com/v1/search?q="
url_list = []

for i in range(len(artists_tracks)):
    song_url = make_url(artists_tracks[i])
    url_list.append(song_url)

url_df = pd.DataFrame(url_list, columns=["url"])
df_all =  pd.concat([df, url_df], axis=1, ignore_index=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df_all.head())


df_all.to_csv('data_library/preprocessing_data/url_data_2000s.csv', index=False, header=False)