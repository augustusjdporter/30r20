import pandas as pd
import requests
import urllib3
from oauthlib.oauth2 import WebApplicationClient
from OAuthBrowser import Chrome, Wait
from urllib.parse import urlparse, parse_qs
from utils import get_secrets

client_id = '116139'
client = WebApplicationClient(client_id)

authorization_url = 'https://www.strava.com/oauth/authorize'

url = client.prepare_request_uri(
  authorization_url,
  redirect_uri = 'http://localhost',
  scope = ['activity:read_all'],
)

def get_strava_data():
    # Initialise browser
    browser = Chrome(window_geometry=(100, 22, 400, 690))
    # Pass Authentication URL
    browser.open_new_window(url)
    # Initialise Wait
    wait = Wait(browser)
    # Wait till query "code" is present in the URL.
    wait.until_present_query('code')
    # Fetch the url
    response_url = urlparse(browser.get_current_url())
    code = parse_qs(response_url.query).get('code')[0]
    print("\nCode: %s\n" % code)
    # Close the browser
    browser.close_current_tab()

    client_secret = get_secrets()["strava_client_secret"]

    ret = requests.post(
        f"https://www.strava.com/oauth/token?client_id=116139&client_secret={client_secret}&code={code}&grant_type=authorization_code")
    print(ret)


    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    auth_url = "https://www.strava.com/oauth/token"
    activites_url = "https://www.strava.com/api/v3/athlete/activities"

    payload = {
        'client_id': "116139",
        'client_secret': client_secret,
        'refresh_token': ret.json()["refresh_token"],
        'grant_type': "refresh_token",
        'f': 'json'
    }

    print("Requesting Token...\n")
    res = requests.post(auth_url, data=payload, verify=False)
    access_token = res.json()['access_token']
    print("Access Token = {}\n".format(access_token))

    header = {'Authorization': 'Bearer ' + access_token}
    param = {'per_page': 200, 'page': 1}
    my_dataset = requests.get(activites_url, headers=header, params=param).json()

    print(my_dataset[0]["name"])
    print(my_dataset[0]["map"]["summary_polyline"])
    df = pd.DataFrame(my_dataset)
    thirty = df[(df["type"] == "Rowing") & (df["elapsed_time"]==1800) & (df["distance"] > 8000)]
    print(thirty)

    thirty = thirty.rename(columns={"start_date_local": "date"})
    return thirty