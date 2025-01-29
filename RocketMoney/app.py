import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, scoped_session
import bcrypt
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.configuration import Configuration
from plaid.api_client import ApiClient
import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
PLAID_CLIENT_ID = os.getenv('PLAID_CLIENT_ID')
PLAID_SECRET = os.getenv('PLAID_SECRET')
PLAID_ENV = os.getenv('PLAID_ENV', 'sandbox')

# Validate Plaid credentials
if not all([PLAID_CLIENT_ID, PLAID_SECRET]):
    raise ValueError("Plaid credentials are missing. Check your environment variables.")

# Plaid client setup
plaid_config = Configuration(
    host=f"https://{PLAID_ENV}.plaid.com",
    api_key={
        "clientId": PLAID_CLIENT_ID,
        "secret": PLAID_SECRET
    }
)
plaid_client = ApiClient(plaid_config)
plaid_api_client = plaid_api.PlaidApi(plaid_client)

# Database setup
Base = declarative_base()
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')
engine = create_engine(DATABASE_URL, echo=False)
Session = scoped_session(sessionmaker(bind=engine))
db_session = Session()

# Define Models
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    access_token = Column(String(256))
    subscriptions = relationship('Subscription', back_populates='user')

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def check_password(self, password):
        return bcrypt.checkpw(password.encode(), self.password_hash.encode())

class Subscription(Base):
    __tablename__ = 'subscriptions'

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(128), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String(128))
    frequency = Column(String(64))
    is_canceled = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    user = relationship('User', back_populates='subscriptions')

# Create tables if not existing
Base.metadata.create_all(engine)

# Helper Functions
def register(username, email, password):
    user = db_session.query(User).filter((User.username == username) | (User.email == email)).first()
    if user:
        return False, "Username or email already exists."
    new_user = User(username=username, email=email)
    new_user.set_password(password)
    db_session.add(new_user)
    db_session.commit()
    return True, "Registration successful!"

def login(username, password):
    user = db_session.query(User).filter(User.username == username).first()
    if user and user.check_password(password):
        return True, user.id
    return False, "Invalid username or password."

def connect_plaid(public_token, user):
    try:
        exchange_request = ItemPublicTokenExchangeRequest(public_token=public_token)
        exchange_response = plaid_api_client.item_public_token_exchange(exchange_request)
        user.access_token = exchange_response.access_token
        db_session.commit()
        return True, "Bank account connected successfully!"
    except Exception as e:
        return False, f"Plaid error: {e}"

def fetch_transactions(user):
    try:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        request = TransactionsGetRequest(
            access_token=user.access_token,
            start_date=start_date,
            end_date=end_date
        )
        response = plaid_api_client.transactions_get(request)
        return response.transactions
    except Exception as e:
        return f"Error fetching transactions: {e}"

# Streamlit UI
def main():
    st.title("RocketMoney Prototype")
    menu = ["Login", "Register"] if not st.session_state.get('authenticated') else ["Dashboard", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register":
        st.subheader("Create a New Account")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            if password != password2:
                st.error("Passwords do not match.")
            else:
                success, message = register(username, email, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    elif choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, user_id = login(username, password)
            if success:
                st.session_state['authenticated'] = True
                st.session_state['user_id'] = user_id
                st.experimental_rerun()
            else:
                st.error(user_id)

    elif choice == "Dashboard":
        user = db_session.query(User).get(st.session_state.get('user_id'))
        st.write(f"Welcome, {user.username}!")
        st.write("Your subscriptions:")
        subscriptions = user.subscriptions
        for sub in subscriptions:
            st.write(f"- {sub.name} (${sub.amount}) - {sub.frequency}")

    elif choice == "Logout":
        st.session_state.clear()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
