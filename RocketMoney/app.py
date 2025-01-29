import streamlit as st
import pandas as pd

# Basic sign-in logic
def authenticate(username, password):
    # Replace this dictionary with your actual authentication logic
    valid_users = {
        "admin": "password123",
        "user1": "mypassword",
    }
    return valid_users.get(username) == password

# App layout
def main():
    st.title("Simple Sign-In and File Upload App")

    # Sign-in section
    st.header("Sign-In")
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.success("Login successful!")
            else:
                st.error("Invalid username or password")
        return

    # File upload section
    st.header("Upload an Excel File")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Read the uploaded Excel file
            df = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("Preview of the uploaded file:")
            st.dataframe(df)  # Display the contents of the file
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Logout option
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()
