import pandas as pd
import requests
from oauthlib.oauth2 import WebApplicationClient
from OAuthBrowser import Chrome, Wait
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from utils import get_secrets
from selenium import webdriver
import time
directory = Path(__file__).parent

client_id = 'PqQOAZNDFYOt5EfrFucgza9gaGBnl0heiBnDoOVF'

client = WebApplicationClient(client_id)

authorization_url = 'https://log.concept2.com/oauth/authorize'

def get_logbook_data():
    url = client.prepare_request_uri(
    authorization_url,
    redirect_uri = 'http://localhost',
    scope = ['results:read'],
    )

    print(url)

    # browser = Chrome(window_geometry=(100, 22, 400, 690))
    # # Pass Authentication URL
    # browser.open_new_window(url)
    # # Initialise Wait
    # wait = Wait(browser)
    # # Wait till query "code" is present in the URL.
    # wait.until_present_query('code')
    driver = webdriver.Edge()
    driver.get(url)
    while True:
        print(driver.current_url)
        current_url = driver.current_url
        if "?code=" in current_url:
            break
        time.sleep(1)

    code = parse_qs(urlparse(current_url).query).get('code')[0]
    print("\nCode: %s\n" % code)
    driver.close()

    client_secret = get_secrets()["logbook_client_secret"]

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    ret = requests.post(url="https://log.concept2.com/oauth/access_token",
                      data={"client_id": client_id,
                            "client_secret": client_secret,
                            "code": code,
                            "grant_type": "authorization_code",
                            "redirect_uri": "http://localhost",
                            "scope": "results:read"},
                      headers=headers)
    access_token = ret.json()['access_token']
    refresh_token = ret.json()['refresh_token']

    header = {'Authorization': 'Bearer ' + access_token}
    print(access_token)
    ret = requests.get(url="https://log.concept2.com/api/users/me/results?from=2023-08-21&to=2023-11-03&type=rower",
                    headers=header).json()
    print(ret)
    ret_df = pd.DataFrame(ret["data"])
    ret_df = ret_df[["date", "distance", "time", "stroke_rate"]]
    thirty = ret_df[(ret_df["time"]==18000) & (ret_df["distance"] >= 8000)]
    print(thirty)
    return thirty
