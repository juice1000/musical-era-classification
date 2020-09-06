import spotipy
from spotipy import util
import subprocess
import json

# client_credentials_manager = SpotifyClientCredentials(client_id='54372e1b817d4af19982352a52541a48', client_secret='ef3cdce4889e4a55aff19bebbdaa5c24')

def dict_iterator(d, song_title):
    stack = list(d.items())
    for k, v in stack:
            if k == "name" and v == song_title:
                #print("%s: %s" % (k, v))
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
    #for item in results['items']:
     #   track = item['track']
      #  print(track['name'] + ' - ' + track['artists'][0]['name'])

    # url = 'https://p.scdn.co/mp3-preview/9a0ddfe0e2598a940f2d87bdfce9a6281137c8ab?cid=774b29d4f13844c495f206cafdad9c86'
    # bearer = 'Bearer %s' %token
    # headers = {'Authorization' : bearer}
    # print(headers)
    # r = requests.get(url, headers=headers)

    auth = "Authorization: Bearer %s" % token

    song_url = "https://api.spotify.com/v1/search?q=abba%20money%20money%20money&type=track&market=US"
    song_title = "Money Money Money"
    json_file = subprocess.check_output(["curl", "-X" ,"GET", song_url, "-H", auth])
    dictionary = json.loads(json_file)
    dictionary = dictionary['tracks']['items']

    download_url = ""

    for dicts in dictionary:
        if dict_iterator(dicts, song_title) is not None:
            download_url = dict_iterator(dicts, song_title)
            break

    if download_url != "":
        subprocess.check_output(["curl", download_url, "-H", auth,"--output", "file1.mp3"])
    else:
        "DOWNLOAD URL COULD NOT BE FETCHED"



else:
    print("Can't get token for", username)

# token = credentials.get_access_token()
# spotify = spotipy.Spotify(auth=token)


# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
