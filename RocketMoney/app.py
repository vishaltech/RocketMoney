import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables if needed
load_dotenv()

# --- USER AUTHENTICATION ---
def authenticate(username, password):
    """Basic authentication function"""
    valid_users = {
        "admin": "password123",
        "user1": "pass456"
    }
    return valid_users.get(username) == password

# --- UI: LOGIN PAGE ---
st.title("🚀 RocketMoney - Secure File Upload")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("Login to continue")
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.success("Login successful! 🎉")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password. Try again.")

# --- UI: EXCEL FILE UPLOAD ---
if st.session_state.logged_in:
    st.subheader("Upload an Excel File 📂")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.write("### Preview of Uploaded Data:")
            st.dataframe(df)  # Display the Excel file contents
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()
