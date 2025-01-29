import os
import streamlit as st
from dotenv import load_dotenv
from plaid.api.plaid_api import PlaidApi
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.api_client import ApiClient
from plaid.configuration import Configuration

# Load environment variables
load_dotenv()

# Plaid API keys
PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID", "679731ac0ef3330026c8b5e9")
PLAID_SECRET = os.getenv("PLAID_SECRET", "7ab1bf5770ed22eb85fb6297824dec")
PLAID_ENV = os.getenv("PLAID_ENV", "sandbox")

# Plaid API environment host
PLAID_HOSTS = {
    "sandbox": "https://sandbox.plaid.com",
    "development": "https://development.plaid.com",
    "production": "https://production.plaid.com",
}
PLAID_HOST = PLAID_HOSTS.get(PLAID_ENV, PLAID_HOSTS["sandbox"])

# Plaid API configuration
config = Configuration(
    host=PLAID_HOST,
    api_key={"clientId": PLAID_CLIENT_ID, "secret": PLAID_SECRET},
)
api_client = ApiClient(config)
plaid_client = PlaidApi(api_client)


# Helper function to create a Link token
def create_link_token():
    user = LinkTokenCreateRequestUser(client_user_id="unique-user-id")
    request = LinkTokenCreateRequest(
        user=user,
        client_name="Rocket Money",
        products=[Products.AUTH],
        country_codes=[CountryCode.US],
        language="en",
    )
    response = plaid_client.link_token_create(request)
    return response["link_token"]


# Streamlit app
def main():
    st.title("Rocket Money with Plaid Integration")

    # Login section
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # Dummy authentication for demonstration
            if username == "admin" and password == "password":
                st.session_state.authenticated = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
    else:
        st.subheader("Plaid Integration")
        if st.button("Generate Link Token"):
            try:
                link_token = create_link_token()
                st.success(f"Link token generated: {link_token}")
                st.write("Use this token to connect with Plaid Link.")
            except Exception as e:
                st.error(f"Failed to generate link token: {e}")

        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.clear()
            st.experimental_rerun()


if __name__ == "__main__":
    main()
