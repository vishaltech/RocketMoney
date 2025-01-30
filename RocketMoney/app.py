import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector
from io import BytesIO
import sqlite3
from sqlalchemy import create_engine
import hashlib
import datetime
import glob
import os
import base64
from graphviz import Digraph
import zipfile

# Page Configuration
st.set_page_config(page_title="🚀 DataForge Pro", layout="wide", page_icon="🔮")
st.title("🧩 DataForge Pro: Multi-Dimensional Analytics")
st.write("""
**Enterprise-Grade Data Fusion Platform**  
*Multi-Source Analysis • Cross-Dataset Querying • Data Lineage • Version Control*
""")

# ======== Global State ========
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'data_versions' not in st.session_state:
    st.session_state.data_versions = {}
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []

# ======== Utility Functions ========
def create_in_memory_db():
    engine = create_engine('sqlite:///:memory:')
    for name, df in st.session_state.datasets.items():
        df.to_sql(name, engine, if_exists='replace', index=False)
    return engine

def log_audit(action):
    timestamp = datetime.datetime.now().isoformat()
    st.session_state.audit_log.append(f"{timestamp} | {action}")

# ======== Sidebar ========
with st.sidebar:
    st.header("⚙️ DataForge Console")
    
    # Multi-File Upload
    uploaded_files = st.file_uploader("📤 Upload Multiple Datasets", 
                                    type=["csv", "xls", "xlsx"],
                                    accept_multiple_files=True)
    
    # Dataset Naming
    for file in uploaded_files:
        if file.name not in st.session_state.datasets:
            default_name = os.path.splitext(file.name)[0][:15]
            dataset_name = st.text_input(f"Name for {file.name}", 
                                       value=default_name,
                                       key=f"name_{file.name}")
            if dataset_name:
                try:
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    st.session_state.datasets[dataset_name] = df
                    st.session_state.data_versions[dataset_name] = [df.copy()]
                    log_audit(f"Dataset Added: {dataset_name}")
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")

    # Advanced Tools
    st.subheader("🔧 DataForge Tools")
    tool_choice = st.selectbox("Select Tool", [
        "Data Lineage Visualizer",
        "Schema Evolution Tracker",
        "Data Checksum Validator",
        "Bulk Data Ops"
    ])

# ======== Main Interface ========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Data Explorer", 
    "🛠 Data Ops", 
    "🔍 SQL Studio", 
    "📜 Audit Trail",
    "🚀 Deployment"
])

with tab1:  # Data Explorer
    st.subheader("🌐 Multi-Dataset Explorer")
    
    # Dataset Selector
    selected_ds = st.selectbox("Choose Dataset", list(st.session_state.datasets.keys()))
    
    if selected_ds:
        df = st.session_state.datasets[selected_ds]
        
        # Advanced DataFrame Profiling
        cols = st.columns(4)
        cols[0].metric("📦 Size", f"{df.memory_usage().sum()/1e6:.2f} MB")
        cols[1].metric("🆔 Checksum", hashlib.md5(df.to_json().encode()).hexdigest()[:8])
        cols[2].metric("⏳ Versions", len(st.session_state.data_versions[selected_ds]))
        cols[3].metric("🔗 Relations", len(df.columns))
        
        # Interactive Schema Viewer
        with st.expander("🔍 Schema Inspector"):
            schema_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Unique Values': df.nunique().values,
                'Null %': (df.isnull().mean()*100).round(2).values
            })
            st.dataframe(schema_df.style.background_gradient())
        
        # Data Preview with Version Compare
        col1, col2 = st.columns(2)
        with col1:
            st.write("Current Version Preview")
            st.dataframe(df.head())
        with col2:
            version_compare = st.selectbox("Compare with Version", 
                                         range(len(st.session_state.data_versions[selected_ds])))
            st.write(f"Version {version_compare} Preview")
            st.dataframe(st.session_state.data_versions[selected_ds][version_compare].head())

with tab2:  # Data Ops
    st.subheader("🛠 Advanced Data Operations")
    
    # Cross-Dataset Joins
    with st.expander("🔗 Data Fusion (Join Datasets)"):
        ds1 = st.selectbox("Primary Dataset", list(st.session_state.datasets.keys()))
        ds2 = st.selectbox("Secondary Dataset", list(st.session_state.datasets.keys()))
        join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"])
        join_key = st.selectbox("Join Key", list(set(st.session_state.datasets[ds1].columns) & 
                                               set(st.session_state.datasets[ds2].columns)))
        if st.button("Fuse Datasets"):
            merged = pd.merge(st.session_state.datasets[ds1], 
                            st.session_state.datasets[ds2], 
                            on=join_key, 
                            how=join_type)
            new_name = f"{ds1}_X_{ds2}"
            st.session_state.datasets[new_name] = merged
            st.session_state.data_versions[new_name] = [merged.copy()]
            log_audit(f"Merged {ds1} with {ds2} as {new_name}")
    
    # Data Version Control
    with st.expander("🕰️ Time Machine"):
        selected_ds = st.selectbox("Dataset", list(st.session_state.datasets.keys()))
        versions = list(range(len(st.session_state.data_versions[selected_ds])))
        selected_version = st.select_slider("Select Version", options=versions)
        if st.button("Restore This Version"):
            st.session_state.datasets[selected_ds] = st.session_state.data_versions[selected_ds][selected_version]
            st.success(f"Restored {selected_ds} to version {selected_version}")

with tab3:  # SQL Studio
    st.subheader("🔍 Cross-Dataset SQL Studio")
    
    # SQL Editor
    query = st.text_area("Write SQL Query", height=200,
                       help="Use dataset names as tables. Example: SELECT * FROM sales JOIN users ON sales.id = users.id")
    
    if st.button("▶️ Execute Query"):
        try:
            engine = create_in_memory_db()
            result = pd.read_sql(query, engine)
            st.write("Query Results")
            st.dataframe(result)
            
            # Visual Query Explainer
            with st.expander("🔍 Query Analysis"):
                cols = st.columns(3)
                cols[0].metric("Result Size", f"{len(result):,} rows")
                cols[1].metric("Columns", len(result.columns))
                cols[2].metric("Memory", f"{result.memory_usage().sum()/1e6:.2f} MB")
                
                st.write("Data Lineage")
                dot = Digraph()
                for table in pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)['name']:
                    dot.node(table)
                dot.edges([(table, 'result') for table in pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)['name']])
                st.graphviz_chart(dot)
        except Exception as e:
            st.error(f"Query Error: {str(e)}")

with tab4:  # Audit Trail
    st.subheader("📜 Data Audit Trail")
    
    # Audit Log Viewer
    st.write("### 🕵️ Activity History")
    for entry in reversed(st.session_state.audit_log[-50:]):
        st.code(entry)
    
    # Data Lineage Visualization
    st.write("### 🔗 System Data Lineage")
    dot = Digraph()
    for ds in st.session_state.datasets:
        dot.node(ds)
    dot.edges([(ds1, ds2) for ds1, ds2 in zip(list(st.session_state.datasets.keys())[:-1], 
                                            list(st.session_state.datasets.keys())[1:])])
    st.graphviz_chart(dot)

with tab5:  # Deployment
    st.subheader("🚀 Enterprise Deployment")
    
    # Snowflake Integration
    with st.expander("❄️ Snowflake Cloud Sync"):
        sf_config = {
            'user': st.text_input("Username"),
            'password': st.text_input("Password", type="password"),
            'account': st.text_input("Account URL"),
            'warehouse': st.text_input("Warehouse"),
            'database': st.text_input("Database"),
            'schema': st.text_input("Schema")
        }
        
        if st.button("☁️ Full Deployment to Snowflake"):
            try:
                conn = snowflake.connector.connect(**sf_config)
                for name, df in st.session_state.datasets.items():
                    df.to_sql(name, conn, if_exists='replace', index=False)
                st.success("All datasets deployed to Snowflake!")
            except Exception as e:
                st.error(f"Deployment Error: {str(e)}")
    
    # Bulk Export
    with st.expander("📤 Export All Data"):
        export_format = st.selectbox("Format", ["ZIP (CSV)", "ZIP (Excel)", "SQLite DB"])
        if st.button("📦 Package Data"):
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for name, df in st.session_state.datasets.items():
                    if export_format == "ZIP (CSV)":
                        zip_file.writestr(f"{name}.csv", df.to_csv())
                    elif export_format == "ZIP (Excel)":
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer) as writer:
                            df.to_excel(writer, sheet_name=name)
                        zip_file.writestr(f"{name}.xlsx", excel_buffer.getvalue())
                    elif export_format == "SQLite DB":
                        db_buffer = BytesIO()
                        engine = create_engine(f'sqlite:///{db_buffer}')
                        df.to_sql(name, engine, index=False)
                        zip_file.writestr("data.db", db_buffer.getvalue())
            st.download_button("Download Package", zip_buffer.getvalue(), "data_package.zip")

# ======== Requirements ========
st.sidebar.divider()
st.sidebar.download_button("📦 Download Requirements", 
                         text="streamlit==1.29.0\npandas==2.1.4\nplotly==5.18.0\nsnowflake-connector-python==3.6.0\nsqlalchemy==2.0.23\ngraphviz==0.20.1\nopenpyxl==3.1.2\nxlsxwriter==3.1.9\npython-magic==0.4.27", 
                         file_name="requirements.txt")
