import streamlit as st
import pandas as pd
import yaml  # Ensure pyyaml is installed
from pandasai.smart_dataframe import SmartDataframe  # Correct import
from pandasai.llm.openai import OpenAI
import os

# Ensure API Key is set correctly
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI
llm = OpenAI(api_key=OPENAI_API_KEY)

# User Authentication Logic
def login_user():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password":  # Dummy check
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"✅ Welcome, {username}! Redirecting...")
            st.rerun()  # Corrected from `st.experimental_rerun()`
        else:
            st.error("❌ Invalid username or password. Try again.")

# Main App Logic
def main():
    st.title("RocketMoney AI Dashboard")
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        option = st.sidebar.radio("Select an option", ["Login", "Register"])
        if option == "Login":
            login_user()
        else:
            st.write("🚧 Registration is under construction.")
    else:
        st.write("💰 Welcome to RocketMoney AI!")
        df = pd.DataFrame({"Category": ["Income", "Expenses", "Savings"], "Amount": [5000, 2000, 1500]})
        sdf = SmartDataframe(df, config={"llm": llm})
        st.write(sdf)

if __name__ == "__main__":
    main()
