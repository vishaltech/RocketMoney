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
from typing import Dict

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

# In-memory user database (for simplicity, use a real database in production)
users_db: Dict[str, Dict] = {}

# Function to create a Plaid Link token
def create_link_token(user_id: str):
    user = LinkTokenCreateRequestUser(client_user_id=user_id)
    request = LinkTokenCreateRequest(
        user=user,
        client_name="Rocket Money",
        products=[Products.AUTH],
        country_codes=[CountryCode.US],
        language="en",
    )
    response = plaid_client.link_token_create(request)
    return response.link_token


# Streamlit app
def main():
    st.title("Rocket Money with Plaid Integration")

    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    # Registration Section
    if not st.session_state.authenticated:
        st.subheader("Register or Login")
        option = st.radio("Choose an option", ["Login", "Register"])

        if option == "Register":
            st.subheader("User Registration")
            username = st.text_input("Enter a username")
            password = st.text_input("Enter a password", type="password")
            if st.button("Register"):
                if username in users_db:
                    st.error("Username already exists!")
                elif username and password:
                    users_db[username] = {"password": password}
                    st.success("User registered successfully!")
                else:
                    st.error("Please provide valid credentials.")

        elif option == "Login":
            st.subheader("User Login")
            username = st.text_input("Enter your username")
            password = st.text_input("Enter your password", type="password")
            if st.button("Login"):
                if username in users_db and users_db[username]["password"] == password:
                    st.session_state.authenticated = True
                    st.session_state.user_id = username
                    st.success("Logged in successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")

    # Dashboard Section
    if st.session_state.authenticated:
        st.subheader(f"Welcome, {st.session_state.user_id}!")
        st.write("You are logged in.")
        
        # Plaid Link token generation
        if st.button("Generate Plaid Link Token"):
            try:
                link_token = create_link_token(st.session_state.user_id)
                st.success("Plaid Link token generated!")
                st.write(f"Link Token: {link_token}")
            except Exception as e:
                st.error(f"Error generating Plaid Link token: {e}")

        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.success("You have been logged out.")
            st.experimental_rerun()


if __name__ == "__main__":
    main()
