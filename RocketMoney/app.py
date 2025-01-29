import os
import streamlit as st
from dotenv import load_dotenv
from plaid.api import plaid_api
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.api_client import ApiClient
from plaid.configuration import Configuration
from urllib3.exceptions import InsecureRequestWarning
import warnings

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Load environment variables
load_dotenv()

# Plaid API keys
PLAID_CLIENT_ID = "679731ac0ef3330026c8b5e9"
PLAID_SECRET = "7ab1bf5770ed22eb85fb6297824dec"
PLAID_ENV = "sandbox"

# Plaid configuration
configuration = Configuration(
    host=plaid_api.Environment.Sandbox,
    api_key={
        "clientId": PLAID_CLIENT_ID,
        "secret": PLAID_SECRET
    }
)

api_client = ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)

# Helper function to create a Link token
def create_link_token():
    user = LinkTokenCreateRequestUser(
        client_user_id="user-unique-id"  # Replace this with a unique identifier for your users
    )
    request = LinkTokenCreateRequest(
        user=user,
        client_name="Rocket Money App",
        products=[Products.AUTH],
        country_codes=[CountryCode.US],
        language="en"
    )
    response = client.link_token_create(request)
    return response.link_token

# Helper function to exchange public token for access token
def exchange_public_token(public_token):
    request = ItemPublicTokenExchangeRequest(
        public_token=public_token
    )
    response = client.item_public_token_exchange(request)
    return response.access_token, response.item_id

# Streamlit app
def main():
    st.title("Rocket Money with Plaid Integration")

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # Dummy authentication
            if username == "admin" and password == "password":
                st.session_state.authenticated = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
    else:
        st.subheader("Plaid Integration")
        st.write("Generate a Plaid Link Token and simulate user bank account integration.")

        if st.button("Create Link Token"):
            try:
                link_token = create_link_token()
                st.success(f"Link Token created: {link_token}")
                st.write("Use the link token to integrate the frontend.")
            except Exception as e:
                st.error(f"Failed to create Link Token: {e}")

        public_token = st.text_input("Public Token", placeholder="Enter public token after connecting with Plaid Link")
        if st.button("Exchange Public Token"):
            try:
                access_token, item_id = exchange_public_token(public_token)
                st.success("Public Token exchanged successfully!")
                st.write(f"Access Token: {access_token}")
                st.write(f"Item ID: {item_id}")
            except Exception as e:
                st.error(f"Failed to exchange public token: {e}")

        if st.button("Logout"):
            st.session_state.authenticated = False
            st.success("Logged out successfully!")
            st.experimental_rerun()

if __name__ == "__main__":
    main()
