import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3  # Ensure SQLite is installed
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
import altair as alt
from fpdf import FPDF

# Streamlit Page Configuration
st.set_page_config(page_title="🚀 SQL Data Analyzer", layout="wide")
st.title("📊 SQL-Powered Data Analyzer")
st.write("Upload an Excel or CSV file to **explore, clean, visualize, and query with SQL!**")

# 🔹 User Input for Table Name
table_name = st.text_input("🔤 Enter the SQL table name:", "uploaded_data")  # Default: uploaded_data
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

        # Display Data Preview
        st.write("### 📝 Data Preview")
        st.dataframe(df.head())

        # Column Selection
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        # 🔥 SQL Query Execution
        st.write(f"### 🛠 Query Your Data with SQL (Table: `{table_name}`)")
        conn = sqlite3.connect(":memory:")  # Use an in-memory database
        df.to_sql(table_name, conn, index=False, if_exists="replace")  # Load into SQL using user-defined name

        query = st.text_area(f"Write your SQL query for `{table_name}` (e.g., `SELECT * FROM {table_name} LIMIT 10;`)")
        if st.button("Run SQL Query"):
            try:
                query_result = pd.read_sql_query(query, conn)
                st.write("### 📊 Query Results")
                st.dataframe(query_result)

                # Allow users to download query results
                query_file = io.BytesIO()
                query_result.to_csv(query_file, index=False)
                query_file.seek(0)
                st.download_button("📥 Download SQL Query Results", query_file, file_name=f"{table_name}_query_results.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ SQL Error: {str(e)}")

        # 📊 Interactive Data Visualization
        st.write("### 📊 Interactive Data Visualization")
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Pie Chart"])
        x_axis = st.selectbox("Select X-axis", numeric_columns + categorical_columns)

        y_axis = None
        if chart_type != "Pie Chart":
            y_axis = st.selectbox("Select Y-axis", numeric_columns)

        if st.button("Generate Chart"):
            if x_axis:
                if chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"📊 Bar Chart: {y_axis} vs {x_axis}")
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"🔬 Scatter Plot: {y_axis} vs {x_axis}")
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis, title=f"📈 Line Chart: {y_axis} vs {x_axis}")
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=x_axis, title=f"🍰 Pie Chart: {x_axis}")
                else:
                    fig = px.histogram(df, x=x_axis, title=f"📊 Histogram: {x_axis}")

                st.plotly_chart(fig)
            else:
                st.warning("⚠️ Please select valid columns for visualization.")

        # 📤 Export Cleaned Data
        st.write("### 📤 Export Cleaned Data")
        cleaned_file = io.BytesIO()
        df.to_csv(cleaned_file, index=False)
        cleaned_file.seek(0)
        st.download_button("📥 Download Cleaned Data", cleaned_file, file_name=f"{table_name}_cleaned_data.csv", mime="text/csv")

    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")
