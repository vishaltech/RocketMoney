import os
from dotenv import load_dotenv
import streamlit as st
from plaid.api.plaid_api import PlaidApi
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.configuration import Configuration
from plaid.api_client import ApiClient

# Load environment variables
load_dotenv()
PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")
PLAID_ENV = os.getenv("PLAID_ENV")

# Configure Plaid client
configuration = Configuration(
    host=f"https://{PLAID_ENV}.plaid.com",
    api_key={
        "clientId": PLAID_CLIENT_ID,
        "secret": PLAID_SECRET,
    }
)
api_client = ApiClient(configuration)
client = PlaidApi(api_client)

# App Title
st.title("RocketMoney Plaid Integration")

# Plaid Link Token Creation
def create_link_token():
    try:
        request = LinkTokenCreateRequest(
            user={"client_user_id": "unique_user_id"},
            client_name="RocketMoney",
            products=[Products("transactions")],
            country_codes=[CountryCode("US")],
            language="en",
        )
        response = client.link_token_create(request)
        return response.link_token
    except Exception as e:
        st.error(f"Error creating link token: {e}")
        return None

# User Interface
if "link_token" not in st.session_state:
    st.session_state["link_token"] = create_link_token()

if st.session_state["link_token"]:
    st.subheader("Connect Your Bank Account")
    st.markdown(f"""
        <a href="https://plaid.com/link/?token={st.session_state['link_token']}" target="_blank" style="color: white; background-color: #0078D4; padding: 10px 15px; text-decoration: none; border-radius: 5px;">Connect Account</a>
    """, unsafe_allow_html=True)

# Transactions Data
def fetch_transactions(access_token):
    try:
        request = TransactionsGetRequest(
            access_token=access_token,
            start_date="2023-01-01",
            end_date="2025-01-01",
        )
        response = client.transactions_get(request)
        transactions = response.transactions
        return transactions
    except Exception as e:
        st.error(f"Error fetching transactions: {e}")
        return []

# Dummy section for access token and transactions display
access_token = st.text_input("Enter Access Token")
if access_token:
    transactions = fetch_transactions(access_token)
    if transactions:
        st.subheader("Your Transactions")
        for transaction in transactions:
            st.write(f"{transaction.date} - {transaction.merchant_name or transaction.name} - ${transaction.amount}")
    else:
        st.write("No transactions found.")
