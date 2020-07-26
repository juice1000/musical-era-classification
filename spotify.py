import sys
import spotipy
from spotipy import util
#client_credentials_manager = SpotifyClientCredentials(client_id='54372e1b817d4af19982352a52541a48', client_secret='ef3cdce4889e4a55aff19bebbdaa5c24')

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
    for item in results['items']:
        track = item['track']
        print(track['name'] + ' - ' + track['artists'][0]['name'])
else:
    print("Can't get token for", username)

#token = credentials.get_access_token()
#spotify = spotipy.Spotify(auth=token)




#sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
