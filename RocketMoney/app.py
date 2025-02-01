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
from ydata_profiling import ProfileReport
import pyarrow as pa
import pyarrow.parquet as pq

# Set page configuration at the very start!
st.set_page_config(page_title="🚀 DataForge Pro", layout="wide", page_icon="🔮")

# -----------------------
# Helper: Safe Rerun
# -----------------------
def safe_rerun():
    """Try to rerun the app; if not available, do nothing."""
    try:
        st.experimental_rerun()
    except Exception:
        pass

# -----------------------
# User Management Helpers
# -----------------------

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                users = json.load(f)
            return users
        except Exception as e:
            st.error(f"Error reading users file: {e}")
            return {}
    else:
        return {}

def save_users(users):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f)
    except Exception as e:
        st.error(f"Error saving users file: {e}")

# -----------------------
# Authentication Function
# -----------------------
def check_auth():
    # Initialize the authentication flag if needed.
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 DataForge Pro Access")

    # Load current users (if any)
    users = load_users()
    
    # If no users exist, force registration.
    if not users:
        st.info("No users registered yet. Please register an account.")
        auth_mode = "Register"
    else:
        auth_mode = st.radio("Choose Authentication Mode", ("Login", "Register"))

    if auth_mode == "Register":
        st.subheader("Register a New Account")
        reg_username = st.text_input("Choose a username", key="reg_username")
        reg_password = st.text_input("Choose an access key", type="password", key="reg_password")
        reg_confirm = st.text_input("Confirm access key", type="password", key="reg_confirm")
        if st.button("Register"):
            if not reg_username or not reg_password:
                st.error("Please provide both username and access key.")
            elif reg_password != reg_confirm:
                st.error("Access key and confirmation do not match.")
            elif reg_username in users:
                st.error("Username already exists. Please choose a different one.")
            else:
                hashed = hashlib.sha256(reg_password.encode()).hexdigest()
                users[reg_username] = hashed
                save_users(users)
                st.success("Registration successful! Please switch to Login mode below.")
                safe_rerun()
                
    if auth_mode == "Login":
        st.subheader("Log In")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Access Key", type="password", key="login_password")
        if st.button("Login"):
            if login_username not in users:
                st.error("User not registered. Please register first.")
            else:
                hashed = hashlib.sha256(login_password.encode()).hexdigest()
                if users[login_username] == hashed:
                    st.session_state.authenticated = True
                    st.session_state.username = login_username
                    safe_rerun()
                else:
                    st.error("Incorrect access key.")
    return False

if not check_auth():
    st.stop()

# -----------------------
# Main App Title & Description
# -----------------------
st.title("🧩 DataForge Pro: Multi-Dimensional Analytics")
st.write("""
**Enterprise-Grade Data Fusion Platform**  
*Multi-Source Analysis • Cross-Dataset Querying • Data Lineage • Version Control*
""")

# -----------------------
# Global State Initialization
# -----------------------
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

# -----------------------
# Utility Functions
# -----------------------
def log_audit(action: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.audit_log.append(f"{timestamp} - {action}")

@st.cache_data
def process_uploaded_file(file, _sample_size=1.0):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.parquet'):
            df = pd.read_parquet(file)
        else:
            # For Excel files (xls, xlsx)
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

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("⚙️ DataForge Console")
    
    with st.expander("📤 Data Upload Settings"):
        sample_size = st.slider("Data Sampling (%)", 1, 100, 100) / 100
        compression = st.selectbox("Compression", ["None", "gzip", "bz2"])  # (Not currently used)
    
    uploaded_files = st.file_uploader("Upload Datasets", 
                                      type=["csv", "xls", "xlsx", "parquet"],
                                      accept_multiple_files=True)
    
    if uploaded_files:
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

# -----------------------
# Main Interface Tabs
# -----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Data Explorer", 
    "🛠 Data Ops", 
    "🔍 SQL Studio", 
    "📜 Audit Trail",
    "🚀 Deployment"
])

# ---------- Tab 1: Data Explorer ----------
with tab1:
    st.subheader("🌐 Multi-Dataset Explorer")
    
    if st.session_state.datasets:
        selected_ds = st.selectbox("Choose Dataset", list(st.session_state.datasets.keys()))
        df = st.session_state.datasets[selected_ds]
        
        # Display basic metrics
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
                'Null %': (df.isnull().mean() * 100).round(2).values
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
                x_axis = st.selectbox("X Axis", df.columns, key="viz_x")
                y_axis = st.selectbox("Y Axis", df.columns, key="viz_y")
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
    else:
        st.info("No datasets loaded yet. Please upload data from the sidebar.")

# ---------- Tab 2: Data Operations ----------
with tab2:
    st.subheader("🛠 Advanced Data Operations")
    
    if st.session_state.datasets:
        selected_ds = st.selectbox("Select Dataset", list(st.session_state.datasets.keys()), key="ds_ops")
        df = st.session_state.datasets[selected_ds]
        
        # Data Cleaning
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
                    if clean_method == "Fill with Mean":
                        fill_value = df[missing_cols].mean()
                    elif clean_method == "Fill with Median":
                        fill_value = df[missing_cols].median()
                    elif clean_method == "Fill with Mode":
                        fill_value = df[missing_cols].mode().iloc[0]
                    else:
                        fill_value = st.text_input("Enter Custom Fill Value")
                    cleaned = df.fillna(fill_value)
                
                if st.button("Apply Cleaning", key="apply_cleaning"):
                    st.session_state.datasets[selected_ds] = cleaned
                    st.session_state.data_versions[selected_ds].append(cleaned.copy())
                    log_audit(f"Data Cleaning: {selected_ds} - {clean_method}")
                    st.success("Data cleaning applied.")
            else:
                st.info("No missing values detected in this dataset.")
            
        # Data Transformations
        with st.expander("🔀 Transformations"):
            transform_type = st.selectbox("Transformation Type",
                                          ["Filter Rows", "Sort Data", "Rename Columns",
                                           "Type Conversion", "Custom Function"])
            
            if transform_type == "Filter Rows":
                filter_col = st.selectbox("Filter Column", df.columns, key="filter_col")
                filter_val = st.text_input("Filter Value", key="filter_val")
                if st.button("Apply Filter", key="apply_filter"):
                    filtered = df[df[filter_col].astype(str) == filter_val]
                    st.session_state.datasets[selected_ds] = filtered
                    st.session_state.data_versions[selected_ds].append(filtered.copy())
                    log_audit(f"Filter Applied on {selected_ds} where {filter_col} == {filter_val}")
                    st.success("Filter applied.")
            
            elif transform_type == "Type Conversion":
                convert_col = st.selectbox("Select Column", df.columns, key="convert_col")
                new_type = st.selectbox("New Data Type", ["str", "int", "float", "datetime", "category"], key="new_type")
                if st.button("Convert Type", key="convert_type"):
                    try:
                        df[convert_col] = df[convert_col].astype(new_type)
                        st.session_state.datasets[selected_ds] = df
                        st.session_state.data_versions[selected_ds].append(df.copy())
                        log_audit(f"Type Conversion on {selected_ds}: {convert_col} to {new_type}")
                        st.success("Type conversion applied.")
                    except Exception as e:
                        st.error(f"Conversion Error: {str(e)}")
            
            elif transform_type == "Custom Function":
                st.write("Enter a code snippet that transforms the dataframe. Make sure to use the variable `df`.")
                custom_code = st.text_area("Pandas Code", help="For example: df = df[df['col'] > 0]")
                if st.button("Execute Code", key="execute_custom"):
                    try:
                        # Use a local namespace to execute the custom code safely
                        local_namespace = {"df": df, "pd": pd, "np": np}
                        exec(custom_code, local_namespace)
                        if "df" in local_namespace:
                            df_transformed = local_namespace["df"]
                            st.session_state.datasets[selected_ds] = df_transformed
                            st.session_state.data_versions[selected_ds].append(df_transformed.copy())
                            log_audit(f"Custom transformation applied on {selected_ds}")
                            st.success("Custom transformation applied.")
                        else:
                            st.error("The code did not return a dataframe named 'df'.")
                    except Exception as e:
                        st.error(f"Execution Error: {str(e)}")
    else:
        st.info("No datasets loaded to operate on.")

# ---------- Tab 3: SQL Studio ----------
with tab3:
    st.subheader("🔍 Cross-Dataset SQL Studio")
    
    query = st.text_area("Write SQL Query", height=200,
                         help="Use dataset names as tables. Example: SELECT * FROM sales JOIN users ON sales.id = users.id")
    
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("▶️ Execute Query", key="execute_query"):
            st.session_state.query_history.insert(0, query)
    with col2:
        if st.session_state.query_history:
            selected_query = st.selectbox("History", st.session_state.query_history[:10], key="query_history")
    
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
                    selected_num = st.selectbox("Numeric Column", num_cols, key="numeric_col")
                    fig = px.histogram(result, x=selected_num)
                    st.plotly_chart(fig)
                
                st.write("#### Statistical Summary")
                st.table(result.describe())
                
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv, "query_results.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Query Error: {str(e)}")
    else:
        st.info("Enter an SQL query to execute.")

# ---------- Tab 4: Audit Trail ----------
with tab4:
    st.subheader("📜 Audit Trail")
    if st.session_state.audit_log:
        st.write("### System Activity Log")
        for log_entry in reversed(st.session_state.audit_log):
            st.code(log_entry)
    else:
        st.write("No audit entries yet.")

# ---------- Tab 5: Enterprise Deployment ----------
with tab5:
    st.subheader("🚀 Enterprise Deployment")
    
    with st.expander("📤 Export Data"):
        export_format = st.selectbox("Export Format", 
                                     ["Parquet", "CSV", "JSON", "Excel", "SQLite"])
        
        if st.button("Generate Export Package", key="generate_export"):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Export each dataset in the chosen format
                for name, df in st.session_state.datasets.items():
                    file_path = os.path.join(tmpdir, f"{name}.{export_format.lower()}")
                    try:
                        if export_format == "Parquet":
                            table = pa.Table.from_pandas(df)
                            pq.write_table(table, file_path)
                        elif export_format == "CSV":
                            df.to_csv(file_path, index=False)
                        elif export_format == "JSON":
                            df.to_json(file_path, orient="records", lines=True)
                        elif export_format == "Excel":
                            df.to_excel(file_path, index=False)
                        elif export_format == "SQLite":
                            engine = create_engine(f'sqlite:///{file_path}')
                            df.to_sql(name, engine, index=False, if_exists='replace')
                    except Exception as exp:
                        st.error(f"Error exporting {name}: {exp}")
                
                # Create a ZIP package of all exported files
                zip_path = os.path.join(tmpdir, "data_package.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, _, files in os.walk(tmpdir):
                        for file in files:
                            if file != "data_package.zip":
                                full_path = os.path.join(root, file)
                                zipf.write(full_path, arcname=file)
                
                with open(zip_path, "rb") as f:
                    st.download_button("Download Package", f.read(), "data_package.zip", key="download_zip")
    st.info("Deployment tools allow you to export all datasets as a single package.")

# -----------------------
# Sidebar Status
# -----------------------
st.sidebar.markdown("---")
st.sidebar.write(f"🖥️ System Status: {len(st.session_state.datasets)} datasets loaded")
