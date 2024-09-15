import streamlit as st
from propelauth_py import init_base_auth, UnauthorizedException
from streamlit.web.server.websocket_headers import _get_websocket_headers
import requests
import config

AUTH_URL = "https://10549518275.propelauthtest.com"
API_KEY = config.PROPELAUTH_API_KEY

class Auth:
    def __init__(self, auth_url, integration_api_key):
        self.auth = init_base_auth(auth_url, integration_api_key)
        self.auth_url = auth_url
        self.integration_api_key = integration_api_key

    def get_user(self):
        access_token = get_access_token()
        if not access_token:
            return None
        try:
            return self.auth.validate_access_token_and_get_user("Bearer " + access_token)
        except UnauthorizedException as err:
            print("Error validating access token", err)
            return None

    def get_account_url(self):
        return self.auth_url + "/account"
    
    def logout(self):
        refresh_token = get_refresh_token()
        if not refresh_token:
            return False
        logout_body = {"refresh_token": refresh_token}
        url = f"{self.auth_url}/api/backend/v1/logout"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.integration_api_key,
        }
        response = requests.post(url, json=logout_body, headers=headers)
        return response.ok


def get_access_token():
    try: 
        return st.context.cookies["__pa_at"]
    except:
        return None

def get_refresh_token():
    try: 
        return st.context.cookies["__pa_rt"]
    except:
        return None

auth = Auth(
    AUTH_URL,
    API_KEY
)
