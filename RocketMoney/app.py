import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_ace import st_ace
from pandasql import sqldf

# Set page configuration and apply custom CSS styling for a modern look
st.set_page_config(page_title="Advanced Data Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f9fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to process uploaded files
def process_file(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    try:
        if ext == ".csv":
            return {"Sheet1": pd.read_csv(uploaded_file)}
        elif ext == ".xlsx":
            # Read all sheets from the Excel file
            return pd.read_excel(uploaded_file, sheet_name=None)
        elif ext == ".parquet":
            return {"Sheet1": pd.read_parquet(uploaded_file)}
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Sidebar: File uploader and sheet selector
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV, Excel, or Parquet file", type=["csv", "xlsx", "parquet"]
)

if uploaded_file:
    data = process_file(uploaded_file)
    if data is None:
        st.stop()
    sheet_names = list(data.keys())
    sheet_selected = st.sidebar.selectbox("Select Sheet", sheet_names)
    df = data[sheet_selected]
else:
    st.info("Please upload a data file to begin.")
    st.stop()

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Explorer", "Visualization", "SQL Query", "Data Tools"]
)

# --- Tab 1: Data Explorer ---
with tab1:
    st.subheader("Data Explorer")
    st.write("Interactively preview your dataset using AgGrid.")
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(enabled=True, paginationAutoPageSize=True)
    gb.configure_side_bar()
    grid_options = gb.build()
    AgGrid(df, gridOptions=grid_options, height=500, theme="streamlit")

# --- Tab 2: Visualization ---
with tab2:
    st.subheader("Visualization")
    st.write("Create interactive charts with Plotly.")
    # Use only numeric columns for visualization
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-Axis", numeric_columns, key="viz_x")
        with col2:
            y_axis = st.selectbox("Select Y-Axis", numeric_columns, key="viz_y")
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Box"])
        if st.button("Generate Chart"):
            try:
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis)
                elif chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis)
                elif chart_type == "Box":
                    fig = px.box(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating chart: {e}")
    else:
        st.write("Not enough numeric columns available for visualization.")

# --- Tab 3: SQL Query ---
with tab3:
    st.subheader("SQL Query")
    st.write("Query your data using SQL. Use `df` as the table name in your query.")
    query = st_ace(
        language="sql",
        theme="monokai",
        key="sql_editor",
        height=200,
        placeholder="e.g., SELECT * FROM df LIMIT 10",
    )
    if st.button("Run Query"):
        try:
            # Use pandasql to run SQL queries on the DataFrame
            result = sqldf(query, {"df": df})
            st.write("Query Result:")
            st.dataframe(result)
        except Exception as e:
            st.error(f"SQL Query Error: {e}")

# --- Tab 4: Data Tools ---
with tab4:
    st.subheader("Data Tools")
    st.write("Perform basic data cleaning and export the cleaned dataset.")
    if st.button("Clean Data"):
        df_clean = df.copy()
        # Drop columns with all missing values and fill numeric columns with their mean
        df_clean = df_clean.dropna(axis=1, how="all")
        num_cols = df_clean.select_dtypes(include=np.number).columns
        for col in num_cols:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        st.write("Cleaned Data Preview:")
        st.dataframe(df_clean.head())
        csv_data = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
