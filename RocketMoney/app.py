import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, relationship
import bcrypt
from plaid.api import plaid_api
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.configuration import Configuration
from plaid.api_client import ApiClient
import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Plaid Configuration
PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")
PLAID_ENV = os.getenv("PLAID_ENV", "sandbox")

configuration = Configuration(
    host=f"https://{PLAID_ENV}.plaid.com",
    api_key={"clientId": PLAID_CLIENT_ID, "secret": PLAID_SECRET},
)
plaid_client = ApiClient(configuration)
plaid_api_client = plaid_api.PlaidApi(plaid_client)

# Database Configuration
Base = declarative_base()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
engine = create_engine(DATABASE_URL, echo=False)
Session = scoped_session(sessionmaker(bind=engine))
db_session = Session()

# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    access_token = Column(String(256))

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def check_password(self, password):
        return bcrypt.checkpw(password.encode(), self.password_hash.encode())


Base.metadata.create_all(engine)


# Helper Functions
def register_user(username, email, password):
    try:
        user = db_session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        if user:
            return False, "Username or email already exists."
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db_session.add(new_user)
        db_session.commit()
        return True, "Registration successful!"
    except Exception as e:
        return False, f"Error: {str(e)}"


def login_user(username, password):
    try:
        user = db_session.query(User).filter(User.username == username).first()
        if user and user.check_password(password):
            return True, user
        return False, "Invalid username or password."
    except Exception as e:
        return False, f"Error: {str(e)}"


def create_link_token(user_id):
    try:
        request = LinkTokenCreateRequest(
            products=["transactions"],
            client_name="RocketMoney Prototype",
            country_codes=["US"],
            language="en",
            user={"client_user_id": str(user_id)},
        )
        response = plaid_api_client.link_token_create(request)
        return response.link_token
    except Exception as e:
        st.error(f"Error creating link token: {str(e)}")
        return None


def exchange_public_token(public_token):
    try:
        request = ItemPublicTokenExchangeRequest(public_token=public_token)
        response = plaid_api_client.item_public_token_exchange(request)
        return response.access_token
    except Exception as e:
        st.error(f"Error exchanging public token: {str(e)}")
        return None


def fetch_transactions(access_token):
    try:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime(
            "%Y-%m-%d"
        )
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        request = TransactionsGetRequest(
            access_token=access_token, start_date=start_date, end_date=end_date
        )
        response = plaid_api_client.transactions_get(request)
        return response.transactions
    except Exception as e:
        st.error(f"Error fetching transactions: {str(e)}")
        return []


# Streamlit Application
def main():
    st.title("RocketMoney Prototype")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_id = None

    menu = ["Login", "Register"] if not st.session_state.authenticated else ["Dashboard"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register":
        st.subheader("Create an Account")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            success, message = register_user(username, email, password)
            if success:
                st.success(message)
            else:
                st.error(message)

    elif choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, user = login_user(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.user_id = user.id
                st.session_state.access_token = user.access_token
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error(user)

    elif choice == "Dashboard":
        user_id = st.session_state.user_id
        user = db_session.query(User).filter(User.id == user_id).first()
        st.subheader(f"Welcome, {user.username}!")

        if not user.access_token:
            st.write("Connect to Plaid:")
            link_token = create_link_token(user.id)
            if link_token:
                st.markdown(
                    f"""
                    <a href="https://plaid.com/link/?token={link_token}" target="_blank">
                        <button>Connect Bank Account</button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("Your Transactions:")
            transactions = fetch_transactions(user.access_token)
            for txn in transactions:
                st.write(f"- {txn.name}: ${txn.amount}")


if __name__ == "__main__":
    main()
