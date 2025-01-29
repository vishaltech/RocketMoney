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
    try:
        user = db_session.query(User).filter((User.username == username) | (User.email == email)).first()
        if user:
            return False, "Username or email already exists."
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db_session.add(new_user)
        db_session.commit()
        return True, "Registration successful!"
    except Exception as e:
        return False, f"Error during registration: {e}"

def login(username, password):
    try:
        user = db_session.query(User).filter(User.username == username).first()
        if user is None:
            return False, "User not found."
        if user.check_password(password):
            return True, user.id
        else:
            return False, "Invalid password."
    except Exception as e:
        return False, f"Error during login: {e}"

def get_user(user_id):
    try:
        return db_session.query(User).filter(User.id == user_id).first()
    except Exception as e:
        return None

# Streamlit UI
def main():
    st.title("RocketMoney Prototype")
    
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['user_id'] = None

    menu = ["Login", "Register"] if not st.session_state['authenticated'] else ["Dashboard", "Logout"]
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
        st.subheader("Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, result = login(username, password)
            if success:
                st.session_state['authenticated'] = True
                st.session_state['user_id'] = result
                st.success("Logged in successfully!")
                st.experimental_set_query_params()  # Ensure clean state
                st.experimental_rerun()
            else:
                st.error(result)

    elif choice == "Dashboard":
        user = get_user(st.session_state['user_id'])
        if user:
            st.subheader(f"Welcome, {user.username}!")
        else:
            st.error("User not found.")

    elif choice == "Logout":
        st.session_state['authenticated'] = False
        st.session_state['user_id'] = None
        st.success("You have been logged out.")
        st.experimental_rerun()

if __name__ == "__main__":
    main()
