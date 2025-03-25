import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import duckdb
import sqlparse
import tempfile
import hashlib
import openai
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_ace import st_ace
from cryptography.fernet import Fernet
from io import BytesIO
from pandas.api.types import is_datetime64_any_dtype

# Configuration
openai.api_key = st.secrets["OPENAI_KEY"]
ENCRYPTION_KEY = Fernet.generate_key()
MAX_FILE_SIZE_MB = 500
ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'parquet', 'feather']

st.set_page_config(layout="wide", page_title="QuantumAnalyzer Pro", page_icon="🚀")

# Security Layer
def encrypt_data(data):
    fernet = Fernet(ENCRYPTION_KEY)
    return fernet.encrypt(data)

def decrypt_data(data):
    fernet = Fernet(ENCRYPTION_KEY)
    return fernet.decrypt(data)

# AI Integration
def generate_query_suggestion(schema):
    prompt = f"Generate 3 advanced SQL queries based on this schema: {schema}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Data Processing
def process_file(uploaded_file):
    file_ext = Path(uploaded_file.name).suffix[1:].lower()
    
    if file_ext == 'csv':
        return {'Sheet1': pd.read_csv(uploaded_file)}
    elif file_ext == 'parquet':
        return {'Sheet1': pd.read_parquet(uploaded_file)}
    elif file_ext == 'xlsx':
        return pd.read_excel(uploaded_file, sheet_name=None)
    return None

# Advanced Visualization
def create_interactive_viz(df):
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox('X Axis', df.columns)
        y_axis = st.selectbox('Y Axis', df.columns)
        chart_type = st.selectbox('Chart Type', 
            ['3D Scatter', 'Heatmap', 'Parallel Categories', 'Sunburst'])
    
    with col2:
        color = st.selectbox('Color Encoding', [None] + list(df.columns))
        animation = st.selectbox('Animation Frame', [None] + list(df.columns))
    
    if chart_type == '3D Scatter':
        fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=color, 
                          animation_frame=animation)
    elif chart_type == 'Heatmap':
        fig = px.density_heatmap(df, x=x_axis, y=y_axis, nbinsx=50, nbinsy=50)
    elif chart_type == 'Parallel Categories':
        fig = px.parallel_categories(df, dimensions=df.columns.tolist()[:5])
    elif chart_type == 'Sunburst':
        fig = px.sunburst(df, path=df.columns.tolist()[:3], values=y_axis)
    
    st.plotly_chart(fig, use_container_width=True)

# SQL IDE Component
def sql_ide(dataframes):
    selected_sheet = st.selectbox("Select Dataset", list(dataframes.keys()))
    df = dataframes[selected_sheet]
    
    col1, col2 = st.columns([1, 1])
    
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
                result = duckdb.query(query).to_df()
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

# Main App
def main():
    st.title("🚀 QuantumAnalyzer Pro - Next-Gen Data Analysis Suite")
    
    uploaded_file = st.file_uploader("Upload Dataset", type=ALLOWED_EXTENSIONS, 
                                   accept_multiple_files=False)
    
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
            return
            
        data = process_file(uploaded_file)
        if not data:
            st.error("Unsupported file format")
            return
            
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "⚡ SQL Lab", 
                                        "📈 Visualize", "🔧 Utilities"])
        
        with tab1:
            sheet_names = list(data.keys())
            selected_sheet = st.selectbox("Select Sheet", sheet_names)
            gb = GridOptionsBuilder.from_dataframe(data[selected_sheet])
            gb.configure_pagination(enabled=True)
            gb.configure_side_bar()
            gb.configure_default_column(groupable=True, value=True, 
                                      enableRowGroup=True, aggFunc='sum')
            grid_options = gb.build()
            AgGrid(data[selected_sheet], gridOptions=grid_options, 
                 height=600, theme='streamlit')
        
        with tab2:
            sql_ide(data)
        
        with tab3:
            create_interactive_viz(data[selected_sheet])
        
        with tab4:
            st.subheader("Advanced Tools")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Data Transformation")
                if st.button("Auto-Clean Data"):
                    # Advanced cleaning logic
                    df = data[selected_sheet]
                    df = df.dropna(axis=1, how='all')
                    df = df.apply(lambda col: col.fillna(col.mean()) 
                               if np.issubdtype(col.dtype, np.number) else col)
                    st.success("Data cleaned automatically!")
                
                st.download_button("Export to Excel", data[selected_sheet].to_excel(),
                                 file_name="exported_data.xlsx")
            
            with col2:
                st.write("### AI Assistant")
                if st.button("Generate Query Suggestions"):
                    schema = str(data[selected_sheet].dtypes)
                    suggestions = generate_query_suggestion(schema)
                    st.markdown(f"```sql\n{suggestions}\n```")

if __name__ == "__main__":
    main()
