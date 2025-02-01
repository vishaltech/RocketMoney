# dataforge_pro.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import hashlib
import datetime
import os
import base64
from graphviz import Digraph
import zipfile
from sqlalchemy import create_engine
import re
import json
import tempfile
from pandas_profiling import ProfileReport
import pyarrow as pa
import pyarrow.parquet as pq

# Security Configuration
ADMIN_PASSWORD = os.getenv("ADMIN_PASS", "DataForge123!")

# Page Configuration
st.set_page_config(page_title="🚀 DataForge Pro", layout="wide", page_icon="🔮")

# ======== Authentication ========
def check_auth():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        with st.container():
            st.title("🔒 DataForge Pro Login")
            password = st.text_input("Enter Access Key:", type="password")
            if st.button("Authenticate"):
                if password == ADMIN_PASSWORD:
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid access key")
        return False
    return True

if not check_auth():
    st.stop()

# Main App
st.title("🧩 DataForge Pro: Multi-Dimensional Analytics")
st.write("""
**Enterprise-Grade Data Fusion Platform**  
*Multi-Source Analysis • Cross-Dataset Querying • Data Lineage • Version Control*
""")

# ======== Enhanced Global State ========
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'data_versions' not in st.session_state:
    st.session_state.data_versions = {}
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []
if 'join_sequence' not in st.session_state:
    st.session_state.join_sequence = []
if 'lineage' not in st.session_state:
    st.session_state.lineage = {}
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# ======== Enhanced Utility Functions ========
@st.cache_data
def process_uploaded_file(file, _sample_size=1.0):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
            
        if _sample_size < 1.0:
            df = df.sample(frac=_sample_size)
        return df
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def create_in_memory_db():
    engine = create_engine('sqlite:///:memory:')
    for name, df in st.session_state.datasets.items():
        df.to_sql(name, engine, if_exists='replace', index=False)
    return engine

def generate_data_profile(df):
    profile = ProfileReport(df, explorative=True)
    return profile.to_html()

# ======== Sidebar ========
with st.sidebar:
    st.header("⚙️ DataForge Console")
    
    with st.expander("📤 Data Upload Settings"):
        sample_size = st.slider("Data Sampling (%)", 1, 100, 100)/100
        compression = st.selectbox("Compression", ["None", "gzip", "bz2"])
    
    uploaded_files = st.file_uploader("Upload Datasets", 
                                    type=["csv", "xls", "xlsx", "parquet"],
                                    accept_multiple_files=True)
    
    for file in uploaded_files:
        try:
            default_name = os.path.splitext(file.name)[0][:15]
            dataset_name = st.text_input(f"Name for {file.name}", 
                                       value=default_name,
                                       key=f"name_{file.name}")
            if dataset_name and dataset_name not in st.session_state.datasets:
                df = process_uploaded_file(file, sample_size)
                if df is not None:
                    st.session_state.datasets[dataset_name] = df
                    st.session_state.data_versions[dataset_name] = [df.copy()]
                    st.session_state.lineage[dataset_name] = []
                    log_audit(f"Dataset Added: {dataset_name} ({len(df)} rows)")
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

# ======== Enhanced Main Interface ========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Data Explorer", 
    "🛠 Data Ops", 
    "🔍 SQL Studio", 
    "📜 Audit Trail",
    "🚀 Deployment"
])

with tab1:
    st.subheader("🌐 Multi-Dataset Explorer")
    
    if st.session_state.datasets:
        selected_ds = st.selectbox("Choose Dataset", list(st.session_state.datasets.keys()))
        df = st.session_state.datasets[selected_ds]
        
        cols = st.columns(4)
        cols[0].metric("📦 Size", f"{df.memory_usage().sum()/1e6:.2f} MB")
        cols[1].metric("🆔 Checksum", hashlib.md5(df.to_json().encode()).hexdigest()[:8])
        cols[2].metric("⏳ Versions", len(st.session_state.data_versions.get(selected_ds, [])))
        cols[3].metric("🔗 Relations", len(df.columns))
        
        with st.expander("🔍 Schema Inspector"):
            schema_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Unique Values': df.nunique().values,
                'Null %': (df.isnull().mean()*100).round(2).values
            })
            st.dataframe(schema_df.style.background_gradient())
            
            if st.button("📊 Generate Data Profile"):
                profile_html = generate_data_profile(df)
                st.components.v1.html(profile_html, height=800, scrolling=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Current Version Preview")
            st.dataframe(df.head())
        
        with col2:
            with st.expander("📈 Advanced Visualization"):
                x_axis = st.selectbox("X Axis", df.columns)
                y_axis = st.selectbox("Y Axis", df.columns)
                chart_type = st.selectbox("Chart Type", 
                                        ["Scatter", "Line", "Bar", "Histogram"])
                
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis)
                elif chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis)
                
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🛠 Advanced Data Operations")
    
    if st.session_state.datasets:
        selected_ds = st.selectbox("Select Dataset", list(st.session_state.datasets.keys()))
        df = st.session_state.datasets[selected_ds]
        
        with st.expander("🧹 Data Cleaning"):
            st.write("### Missing Value Handling")
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                clean_method = st.selectbox("Treatment Method", 
                                          ["Drop NA", "Fill with Mean", "Fill with Median", 
                                           "Fill with Mode", "Custom Value"])
                if clean_method == "Drop NA":
                    cleaned = df.dropna()
                else:
                    fill_value = None
                    if clean_method == "Fill with Mean":
                        fill_value = df[missing_cols].mean()
                    elif clean_method == "Fill with Median":
                        fill_value = df[missing_cols].median()
                    elif clean_method == "Fill with Mode":
                        fill_value = df[missing_cols].mode().iloc[0]
                    else:
                        fill_value = st.text_input("Enter Custom Fill Value")
                    
                    cleaned = df.fillna(fill_value)
                
                if st.button("Apply Cleaning"):
                    st.session_state.datasets[selected_ds] = cleaned
                    st.session_state.data_versions[selected_ds].append(cleaned.copy())
                    log_audit(f"Data Cleaning: {selected_ds} - {clean_method}")
            
        with st.expander("🔀 Transformations"):
            transform_type = st.selectbox("Transformation Type",
                                        ["Filter Rows", "Sort Data", "Rename Columns",
                                         "Type Conversion", "Custom Function"])
            
            if transform_type == "Filter Rows":
                filter_col = st.selectbox("Filter Column", df.columns)
                filter_val = st.text_input("Filter Value")
                filtered = df[df[filter_col].astype(str) == filter_val]
                if st.button("Apply Filter"):
                    st.session_state.datasets[selected_ds] = filtered
                    st.session_state.data_versions[selected_ds].append(filtered.copy())
            
            elif transform_type == "Type Conversion":
                convert_col = st.selectbox("Select Column", df.columns)
                new_type = st.selectbox("New Data Type",
                                      ["str", "int", "float", "datetime", "category"])
                df[convert_col] = df[convert_col].astype(new_type)
                if st.button("Convert Type"):
                    st.session_state.datasets[selected_ds] = df
                    st.session_state.data_versions[selected_ds].append(df.copy())
            
            elif transform_type == "Custom Function":
                custom_code = st.text_area("Enter Pandas Code (use 'df' as dataframe)")
                if st.button("Execute Code"):
                    try:
                        exec(f"global df; df = {custom_code}")
                        st.session_state.datasets[selected_ds] = df
                        st.session_state.data_versions[selected_ds].append(df.copy())
                    except Exception as e:
                        st.error(f"Execution Error: {str(e)}")

with tab3:
    st.subheader("🔍 Cross-Dataset SQL Studio")
    
    query = st.text_area("Write SQL Query", height=200,
                       help="Use dataset names as tables. Example: SELECT * FROM sales JOIN users ON sales.id = users.id")
    
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("▶️ Execute Query"):
            st.session_state.query_history.insert(0, query)
    with col2:
        selected_query = st.selectbox("History", st.session_state.query_history[:10])
    
    if query:
        try:
            engine = create_in_memory_db()
            result = pd.read_sql(query, engine)
            
            st.write("### Query Results")
            st.dataframe(result)
            
            with st.expander("🔍 Advanced Analysis"):
                st.write("#### Data Distribution")
                num_cols = result.select_dtypes(include=np.number).columns
                if len(num_cols) > 0:
                    selected_num = st.selectbox("Numeric Column", num_cols)
                    fig = px.histogram(result, x=selected_num)
                    st.plotly_chart(fig)
                
                st.write("#### Statistical Summary")
                st.table(result.describe())
                
                csv = result.to_csv(index=False)
                st.download_button("Download Results", csv, "query_results.csv")
                
        except Exception as e:
            st.error(f"Query Error: {str(e)}")

with tab5:
    st.subheader("🚀 Enterprise Deployment")
    
    with st.expander("📤 Export Data"):
        export_format = st.selectbox("Export Format", 
                                   ["Parquet", "CSV", "JSON", "Excel", "SQLite"])
        
        if st.button("Generate Export Package"):
            with tempfile.TemporaryDirectory() as tmpdir:
                for name, df in st.session_state.datasets.items():
                    path = os.path.join(tmpdir, f"{name}.{export_format.lower()}")
                    if export_format == "Parquet":
                        pq.write_table(pa.Table.from_pandas(df), path)
                    elif export_format == "CSV":
                        df.to_csv(path, index=False)
                    elif export_format == "JSON":
                        df.to_json(path, orient="records")
                    elif export_format == "Excel":
                        df.to_excel(path, index=False)
                    elif export_format == "SQLite":
                        engine = create_engine(f'sqlite:///{path}')
                        df.to_sql(name, engine, index=False)
                
                zip_path = os.path.join(tmpdir, "data_package.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, _, files in os.walk(tmpdir):
                        for file in files:
                            if file != "data_package.zip":
                                zipf.write(os.path.join(root, file), file)
                
                with open(zip_path, "rb") as f:
                    st.download_button("Download Package", f.read(), "data_package.zip")

st.sidebar.markdown("---")
st.sidebar.write(f"🖥️ System Status: {len(st.session_state.datasets)} datasets loaded")
