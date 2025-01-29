import streamlit as st
from plaid.api import plaid_api
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
import datetime

# Plaid client setup
PLAID_CLIENT_ID = "679731ac0ef3330026c8b5e9"
PLAID_SECRET = "7ab1bf5770ed22eb85fb6297824dec"
PLAID_ENV = "sandbox"

configuration = plaid.Configuration(
    host=plaid.Environment.Sandbox,
    api_key={
        "clientId": PLAID_CLIENT_ID,
        "secret": PLAID_SECRET,
    },
)
api_client = plaid.ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)

# Function to create a Plaid link token
def create_link_token():
    request = LinkTokenCreateRequest(
        products=[Products.TRANSACTIONS],
        client_name="RocketMoney",
        country_codes=[CountryCode.US],
        language="en",
        user={"client_user_id": st.session_state["user_id"]},
    )
    try:
        response = client.link_token_create(request)
        return response.link_token
    except Exception as e:
        st.error(f"Error creating link token: {e}")
        return None

# Function to exchange public token for access token
def exchange_public_token(public_token):
    request = ItemPublicTokenExchangeRequest(public_token=public_token)
    try:
        response = client.item_public_token_exchange(request)
        return response.access_token
    except Exception as e:
        st.error(f"Error exchanging public token: {e}")
        return None

# Function to fetch transactions
def fetch_transactions(access_token):
    start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    request = TransactionsGetRequest(
        access_token=access_token,
        start_date=start_date,
        end_date=end_date,
    )
    try:
        response = client.transactions_get(request)
        return response.transactions
    except Exception as e:
        st.error(f"Error fetching transactions: {e}")
        return []

# Streamlit app
def main():
    st.title("RocketMoney Prototype")
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = "vishal84520"  # Example user_id
        st.session_state["access_token"] = None

    if st.session_state["authenticated"]:
        st.subheader(f"Welcome, {st.session_state['user_id']}!")

        # Check if access token exists
        if st.session_state["access_token"]:
            transactions = fetch_transactions(st.session_state["access_token"])
            st.subheader("Your Transactions")
            for tx in transactions:
                st.write(f"{tx['date']}: {tx['name']} - ${tx['amount']}")
        else:
            st.warning("No access token found. Connect to Plaid first.")
            if st.button("Connect to Plaid"):
                link_token = create_link_token()
                if link_token:
                    st.write("Link token created successfully. Use it to initialize Plaid Link.")
                    st.write(link_token)

        if st.button("Logout"):
            st.session_state.clear()
            st.success("Logged out successfully!")
            st.rerun()

    else:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            # Simulate login success
            if username == "admin" and password == "password":
                st.session_state["authenticated"] = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials!")

if __name__ == "__main__":
    main()
