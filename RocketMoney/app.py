import streamlit as st
import pandas as pd

def main():
    st.title("File Upload and Display")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['xls', 'xlsx', 'csv'])

    if uploaded_file is not None:
        # Get the file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()

        try:
            if file_extension == 'xlsx':
                # Use openpyxl engine for .xlsx files
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_extension == 'xls':
                # Use xlrd engine for .xls files
                df = pd.read_excel(uploaded_file, engine='xlrd')
            elif file_extension == 'csv':
                # Use pandas read_csv for .csv files
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file type.")
                return

            st.success("File uploaded and read successfully!")
            st.dataframe(df)  # Display the dataframe

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
