import streamlit as st
import pandas as pd
import os
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="RocketMoney - Secure Portal", layout="centered")

# --- Helper Functions ---
def hash_password(password):
    """Hashes password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Loads user credentials from a CSV file."""
    if not os.path.exists("users.csv"):
        return {}
    df = pd.read_csv("users.csv")
    return dict(zip(df.username, df.password))  # Convert to {username: hashed_password}

def save_user(username, password):
    """Saves a new user to CSV file."""
    hashed_pw = hash_password(password)
    df = pd.DataFrame([[username, hashed_pw]], columns=["username", "password"])
    df.to_csv("users.csv", mode="a", header=not os.path.exists("users.csv"), index=False)

def authenticate(username, password):
    """Validates username & password."""
    users = load_users()
    return users.get(username) == hash_password(password)

# --- Initialize Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- USER REGISTRATION ---
def register_user():
    st.title("📝 Register for RocketMoney")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if not new_username or not new_password or not confirm_password:
            st.error("⚠️ All fields are required!")
        elif new_password != confirm_password:
            st.error("❌ Passwords do not match!")
        elif new_username in load_users():
            st.error("🚫 Username already exists!")
        else:
            save_user(new_username, new_password)
            st.success("✅ Registration successful! Please login.")
            st.session_state.page = "Login"
            st.rerun()

# --- LOGIN SYSTEM ---
def login_user():
    st.title("🔑 RocketMoney - Secure Login")
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"✅ Welcome, {username}! Redirecting...")
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Try again.")

# --- FILE UPLOAD INTERFACE ---
def file_upload():
    st.title(f"📂 Welcome, {st.session_state.username} - Upload Your Excel File")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.success("✅ File uploaded successfully!")
            st.write("### Preview of Uploaded Data:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    if st.button("Logout 🔒"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# --- MAIN LOGIC ---
if "page" not in st.session_state:
    st.session_state.page = "Login"

if not st.session_state.logged_in:
    option = st.sidebar.radio("Select an option", ["Login", "Register"])
    if option == "Login":
        login_user()
    else:
        register_user()
else:
    file_upload()
