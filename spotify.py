import spotipy
from spotipy import util
import subprocess
import json
import pandas as pd
import os
import time
import sys
# client_credentials_manager = SpotifyClientCredentials(client_id='54372e1b817d4af19982352a52541a48', client_secret='ef3cdce4889e4a55aff19bebbdaa5c24')



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


# Load data
data = pd.read_csv("data_library/MSD_Artists_300.csv")


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
    #for item in results['items']:
     #   track = item['track']
      #  print(track['name'] + ' - ' + track['artists'][0]['name'])

    # url = 'https://p.scdn.co/mp3-preview/9a0ddfe0e2598a940f2d87bdfce9a6281137c8ab?cid=774b29d4f13844c495f206cafdad9c86'
    # bearer = 'Bearer %s' %token
    # headers = {'Authorization' : bearer}
    # print(headers)
    # r = requests.get(url, headers=headers)
    df = pd.read_csv("data_library/url_data.csv")
    df = df.values
    #print(df)

    counter = 0
    #subprocess.call(['curl', 'https://p.scdn.co/mp3-preview/3205d45d73576078c8c31d55e8691c1c31aa40f2?cid=54372e1b817d4af19982352a52541a48', '-H', 'Authorization: Bearer BQAmTF06iTr3Q64p4BUkyLgwTMfaHMGxQDjgH6L1dyi7iQIk2m6eAt0Fyt3auldOSbkxIecNDE6BaXK1EKrzlzmR5zqgF6XDVqEdJT3W-ofiM942LGrDhbi48aGcAImzrGKwjzL2rATXZy6ah1Lg9TZfyw-g', '--output', 'music_library/Frankie Yankovic/Walter Ostanek -- Blue Skirt Waltz.mp3'])
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

            if os.path.exists("music_library"):
                path = "music_library/" + file_title + ".mp3"
            else:
                os.mkdir("music_library")
                path = "music_library/" + file_title + ".mp3"


            if download_url != "":
                subprocess.call(["curl", download_url, "-H", auth, "--output", path])
            else:
                "DOWNLOAD URL COULD NOT BE FETCHED"

            counter += 1
            print(counter)
        else:
            print("ERROR RETRIEVING URL...")
            time.sleep(1)

else:
    print("Can't get token for", username)

# token = credentials.get_access_token()
# spotify = spotipy.Spotify(auth=token)


# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
