import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import io
import openpyxl
import snowflake.connector

# Streamlit Page Configuration
st.set_page_config(page_title="🚀 SQL & Snowflake Data Analyzer", layout="wide")
st.title("📊 SQL & Snowflake-Powered Data Analyzer")
st.write("Upload an Excel or CSV file to **explore, clean, visualize, query with SQL, and push data to Snowflake!**")

# 🔹 User Input for Table Name
table_name = st.text_input("🔤 Enter the SQL table name:", "uploaded_data")  
table_name = table_name.replace(" ", "_")  # Ensure SQL-safe table names

# File Uploader
uploaded_file = st.file_uploader("📂 Upload your Excel or CSV file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Convert column names to SQL-friendly format
        df.columns = [col.replace(" ", "_") for col in df.columns]

        # Infer data types
        data_types = df.dtypes

        # Display Data Preview
        st.write("### 📝 Data Preview")
        st.dataframe(df.head())

        # Column Selection for Adding Data
        st.write("### ➕ Add New Data")
        selected_column = st.selectbox("Select a column to add data", df.columns)
        selected_data_type = data_types[selected_column]

        # Input for the selected field
        if selected_data_type in ["int64", "float64"]:
            new_value = st.number_input(f"Enter value for {selected_column}")
        else:
            new_value = st.text_input(f"Enter value for {selected_column}")

        if st.button("Add Row"):
            new_row = pd.DataFrame([{selected_column: new_value}])
            df = pd.concat([df, new_row], ignore_index=True)
            st.success("✅ New data added!")
            st.dataframe(df.tail())

        # Export Updated Data to Excel
        st.write("### 📤 Export Updated Data to Excel")
        excel_file = io.BytesIO()
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Updated_Data")
        excel_file.seek(0)
        st.download_button("📥 Download Excel File", excel_file, file_name=f"{table_name}_updated.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Push to Snowflake
        st.write("### ❄️ Push Data to Snowflake")
        snowflake_account = st.text_input("🔑 Snowflake Account (e.g., xyz123.snowflakecomputing.com)")
        snowflake_user = st.text_input("👤 Snowflake Username")
        snowflake_password = st.text_input("🔒 Snowflake Password", type="password")
        snowflake_warehouse = st.text_input("🏢 Warehouse")
        snowflake_database = st.text_input("📂 Database")
        snowflake_schema = st.text_input("📑 Schema")

        if st.button("Upload to Snowflake"):
            try:
                conn = snowflake.connector.connect(
                    user=snowflake_user,
                    password=snowflake_password,
                    account=snowflake_account,
                    warehouse=snowflake_warehouse,
                    database=snowflake_database,
                    schema=snowflake_schema
                )
                cur = conn.cursor()

                # Create table if not exists
                create_table_sql = f"""
                CREATE OR REPLACE TABLE {table_name} ({', '.join(f"{col} STRING" for col in df.columns)});
                """
                cur.execute(create_table_sql)

                # Insert data into Snowflake
                for _, row in df.iterrows():
                    values = ", ".join([f"'{str(value)}'" for value in row.values])
                    insert_sql = f"INSERT INTO {table_name} VALUES ({values});"
                    cur.execute(insert_sql)

                conn.commit()
                st.success(f"✅ Data successfully uploaded to Snowflake table `{table_name}`!")

            except Exception as e:
                st.error(f"❌ Snowflake Error: {str(e)}")

    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")
