import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import duckdb
import sqlparse
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_ace import st_ace
from cryptography.fernet import Fernet
from io import BytesIO

# Configuration
MAX_FILE_SIZE_MB = 500
ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'parquet']

st.set_page_config(layout="wide", page_title="DataAnalyzer Pro", page_icon="🚀")

# Security Layer (for potential future use)
ENCRYPTION_KEY = Fernet.generate_key()

def encrypt_data(data):
    fernet = Fernet(ENCRYPTION_KEY)
    return fernet.encrypt(data)

def decrypt_data(data):
    fernet = Fernet(ENCRYPTION_KEY)
    return fernet.decrypt(data)

# Data Processing
def process_file(uploaded_file):
    file_ext = Path(uploaded_file.name).suffix[1:].lower()
    try:
        if file_ext == 'csv':
            return {'Sheet1': pd.read_csv(uploaded_file)}
        elif file_ext == 'parquet':
            return {'Sheet1': pd.read_parquet(uploaded_file)}
        elif file_ext == 'xlsx':
            return pd.read_excel(uploaded_file, sheet_name=None)
        return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Advanced Visualization
def create_interactive_viz(df):
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox('X Axis', df.columns)
        y_axis = st.selectbox('Y Axis', df.columns)
        chart_type = st.selectbox('Chart Type', 
            ['Scatter', 'Line', 'Bar', 'Histogram', 'Box'])
    
    with col2:
        color = st.selectbox('Color Encoding', [None] + list(df.columns))
        hover_data = st.multiselect('Hover Data', df.columns)
    
    try:
        if chart_type == 'Scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color, hover_data=hover_data)
        elif chart_type == 'Line':
            fig = px.line(df, x=x_axis, y=y_axis, color=color)
        elif chart_type == 'Bar':
            fig = px.bar(df, x=x_axis, y=y_axis, color=color)
        elif chart_type == 'Histogram':
            fig = px.histogram(df, x=x_axis, color=color)
        elif chart_type == 'Box':
            fig = px.box(df, x=x_axis, y=y_axis, color=color)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

# SQL IDE Component
def sql_ide(dataframes, selected_sheet):
    df = dataframes[selected_sheet]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SQL Editor")
        query = st_ace(
            language='sql',
            placeholder="SELECT * FROM df",
            key="sql_editor",
            height=300,
            theme="dracula",
            font_size=14,
            wrap=True
        )
        
        if st.button("▶ Execute Query"):
            try:
                # Pass the dataframe as a parameter to DuckDB
                result = duckdb.query(query, {'df': df}).to_df()
                st.session_state.last_result = result
                st.success("Query executed successfully!")
            except Exception as e:
                st.error(f"Query Error: {str(e)}")
    
    with col2:
        st.subheader("Data Preview")
        AgGrid(df.head(1000), height=300, fit_columns_on_grid_load=True)
        
        if 'last_result' in st.session_state:
            st.subheader("Query Result")
            AgGrid(st.session_state.last_result, height=300)

def main():
    st.title("🚀 DataAnalyzer Pro - Advanced Data Analysis Suite")
    
    uploaded_file = st.file_uploader("Upload Dataset", type=ALLOWED_EXTENSIONS)
    
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
            return
            
        data = process_file(uploaded_file)
        if not data:
            return
        
        # Global sheet selection placed in the sidebar
        sheet_names = list(data.keys())
        selected_sheet = st.sidebar.selectbox("Select Sheet", sheet_names)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Explorer", "⚡ SQL Lab", "📈 Visualize", "🔧 Tools"])
        
        with tab1:
            st.subheader("Data Explorer")
            gb = GridOptionsBuilder.from_dataframe(data[selected_sheet])
            gb.configure_pagination(enabled=True)
            gb.configure_side_bar()
            gb.configure_default_column(groupable=True, enableRowGroup=True)
            grid_options = gb.build()
            AgGrid(data[selected_sheet], gridOptions=grid_options, height=600, theme='streamlit')
        
        with tab2:
            sql_ide(data, selected_sheet)
        
        with tab3:
            create_interactive_viz(data[selected_sheet])
        
        with tab4:
            st.subheader("Data Tools")
            if st.button("Clean Missing Values"):
                df = data[selected_sheet].copy()
                df = df.dropna(axis=1, how='all')
                num_cols = df.select_dtypes(include=np.number).columns
                df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                st.success("Basic cleaning applied!")
            
            st.download_button(
                "Export to CSV",
                data[selected_sheet].to_csv().encode('utf-8'),
                file_name="exported_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
