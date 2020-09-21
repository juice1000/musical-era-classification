import spotipy
from spotipy import util
import subprocess
import json
import pandas as pd
import os
import time


def dict_iterator(d, song_title):
    stack = list(d.items())
    for k, v in stack:
            if k == "name" and v == song_title:
                return(url_check(stack))


def url_check(d):
    for i in d:
        if i[0] == "preview_url" and i[1] is not None:
            preview_url = i[1]
            return preview_url


# Search for name
scope = 'user-library-read'
username = 'Julien Look'
token = util.prompt_for_user_token(username,
                                   scope,
                                   client_id='54372e1b817d4af19982352a52541a48',
                                   client_secret='ef3cdce4889e4a55aff19bebbdaa5c24',
                                   redirect_uri='http://localhost:9090')


if token:
    sp = spotipy.Spotify(auth=token)
    results = sp.current_user_saved_tracks()
    auth = "Authorization: Bearer %s" % token

    df = pd.read_csv("data_library/url_data.csv")
    df_arr = df.values
    checkout = []
    df = df.values

    counter = 0
    for i in range(len(df)):

        song_url = df[i][4]
        song_title = df[i][3]
        file_title = df[i][2] + " -- " + df[i][3]
        json_file = subprocess.check_output(["curl", "-X", "GET", song_url, "-H", auth])
        print(json_file)
        if str(json_file) != "b''" and str(json_file).find("Error") == -1 and str(json_file).find("error") == -1 :
            dictionary = json.loads(json_file)
            dictionary = dictionary['tracks']['items']

            download_url = ""

            for dicts in dictionary:
                if dict_iterator(dicts, song_title) is not None:
                    download_url = dict_iterator(dicts, song_title)
                    break

            if not os.path.exists("music_library"):
                os.mkdir("music_library")

            path = "music_library/" + file_title + ".wav"

            if download_url != "":
                subprocess.call(["curl", download_url, "-H", auth, "--output", path])
                checkout.append(df_arr[i])

            else:
                "DOWNLOAD URL COULD NOT BE FETCHED"

            counter += 1
            print(counter)
        else:
            print("ERROR RETRIEVING URL...")
            time.sleep(1)

    checkout = pd.DataFrame(checkout)
    checkout.to_csv('data_library/list_downloaded_previews.csv', index=False, header=False)

else:
    print("Can't get token for", username)

