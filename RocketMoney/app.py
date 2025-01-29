
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, scoped_session
import bcrypt
from plaid import Client
import datetime
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Configuration
SECRET_KEY = os.getenv('SECRET_KEY') or 'default_secret_key'
PLAID_CLIENT_ID = os.getenv('PLAID_CLIENT_ID')
PLAID_SECRET = os.getenv('PLAID_SECRET')
PLAID_ENV = os.getenv('PLAID_ENV') or 'sandbox'

# Initialize Plaid client
plaid_client = Client(client_id=PLAID_CLIENT_ID,
                      secret=PLAID_SECRET,
                      environment=PLAID_ENV)

# Database setup
Base = declarative_base()
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')
engine = create_engine(DATABASE_URL, echo=True)
Session = scoped_session(sessionmaker(bind=engine))
db_session = Session()

# Define Models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    access_token = Column(String(256))  # Increased length for security tokens
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

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Streamlit Session State for Authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None

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
    finally:
        db_session.close()

def login(username, password):
    try:
        user = db_session.query(User).filter(User.username == username).first()
        if user and user.check_password(password):
            return True, user.id
        return False, "Invalid username or password."
    except Exception as e:
        return False, f"Error during login: {e}"
    finally:
        db_session.close()

def get_user(user_id):
    try:
        return db_session.query(User).filter(User.id == user_id).first()
    except Exception as e:
        return None
    finally:
        db_session.close()

def connect_plaid(user, public_token):
    try:
        exchange_response = plaid_client.Item.public_token.exchange(public_token)
        access_token = exchange_response['access_token']
        user.access_token = access_token
        db_session.commit()
        return True, "Bank account connected successfully!"
    except Exception as e:
        return False, f"Error connecting bank account: {e}"
    finally:
        db_session.close()

def fetch_transactions(user):
    try:
        access_token = user.access_token
        if not access_token:
            return [], [], []
        
        accounts_response = plaid_client.Accounts.get(access_token)
        accounts = accounts_response['accounts']
        
        start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        transactions_response = plaid_client.Transactions.get(access_token, start_date, end_date)
        transactions = transactions_response['transactions']
        
        for txn in transactions:
            if txn.get('recurring'):
                existing_sub = db_session.query(Subscription).filter_by(transaction_id=txn['transaction_id'], user_id=user.id).first()
                if not existing_sub:
                    frequency = determine_frequency(txn)
                    new_sub = Subscription(
                        transaction_id=txn['transaction_id'],
                        name=txn['name'],
                        amount=txn['amount'],
                        category=','.join(txn['category']) if txn['category'] else 'Uncategorized',
                        frequency=frequency,
                        user=user
                    )
                    db_session.add(new_sub)
        db_session.commit()
        subscriptions = db_session.query(Subscription).filter_by(user_id=user.id).all()
        return accounts, transactions, subscriptions
    except Exception as e:
        return [], [], f"Error fetching transactions: {e}"
    finally:
        db_session.close()

def determine_frequency(txn):
    return 'Monthly'

def cancel_subscription(subscription_id, user_id):
    try:
        subscription = db_session.query(Subscription).filter_by(id=subscription_id, user_id=user_id).first()
        if subscription and not subscription.is_canceled:
            subscription.is_canceled = True
            db_session.commit()
            return True, f"Subscription '{subscription.name}' has been canceled."
        elif subscription and subscription.is_canceled:
            return False, f"Subscription '{subscription.name}' is already canceled."
        else:
            return False, "Subscription not found."
    except Exception as e:
        return False, f"Error canceling subscription: {e}"
    finally:
        db_session.close()

def main():
    if os.path.exists('static/styles.css'):
        with open('static/styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    st.title("RocketMoney Prototype")
    menu = ["Login", "Register"]
    if st.session_state['authenticated']:
        menu = ["Dashboard", "Subscriptions", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Register":
        st.subheader("Create a New Account")
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            password2 = st.text_input("Confirm Password", type='password')
            submit = st.form_submit_button("Register")
        
        if submit:
            if password != password2:
                st.error("Passwords do not match.")
            elif not username or not email or not password:
                st.error("Please fill out all fields.")
            else:
                success, message = register(username, email, password)
                if success:
                    st.success(message)
                    st.info("Go to the Login page to log in.")
                else:
                    st.error(message)
    
    elif choice == "Login":
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            submit = st.form_submit_button("Login")
        
        if submit:
            success, result = login(username, password)
            if success:
                st.session_state['authenticated'] = True
                st.session_state['user_id'] = result
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error(result)
    
    elif choice == "Dashboard":
        user = get_user(st.session_state['user_id'])
        st.subheader(f"Welcome, {user.username}!" if user else "User not found.")
    
    elif choice == "Logout":
        st.session_state['authenticated'] = False
        st.session_state['user_id'] = None
        st.success("You have been logged out.")
        st.experimental_rerun()

if __name__ == "__main__":
    main()
