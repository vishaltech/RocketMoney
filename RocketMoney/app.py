import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import hashlib
import datetime
import os
import base64
from graphviz import Digraph
import zipfile
from sqlalchemy import create_engine

# ============ PAGE CONFIGURATION ============
st.set_page_config(page_title="🚀 DataForge Pro", layout="wide", page_icon="🔮")
st.title("🧩 DataForge Pro: Multi-Dimensional Analytics")
st.write("""
**Enterprise-Grade Data Fusion Platform**  
*Multi-Source Analysis • Cross-Dataset Querying • Data Lineage • Version Control*
""")

# ============ GLOBAL STATE ============
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'data_versions' not in st.session_state:
    st.session_state.data_versions = {}
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []
if 'join_sequence' not in st.session_state:
    st.session_state.join_sequence = []

# ============ UTILITY FUNCTIONS ============
def create_in_memory_db():
    """Create SQLite in-memory database with all datasets"""
    engine = create_engine('sqlite:///:memory:')
    for name, df in st.session_state.datasets.items():
        df.to_sql(name, engine, if_exists='replace', index=False)
    return engine

def log_audit(action):
    """Track all system activities"""
    timestamp = datetime.datetime.now().isoformat()
    st.session_state.audit_log.append(f"{timestamp} | {action}")

def get_common_columns(df1, df2):
    """Find matching columns between two dataframes"""
    return list(set(df1.columns) & set(df2.columns))

# ============ SIDEBAR ============
with st.sidebar:
    st.header("⚙️ DataForge Console")
    
    # Multi-File Upload with Dynamic Naming
    uploaded_files = st.file_uploader(
        "📤 Upload Multiple Datasets", 
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True
    )
    
    # Process uploaded files
    for file in uploaded_files:
        if file.name not in st.session_state.datasets:
            default_name = os.path.splitext(file.name)[0][:15]
            dataset_name = st.text_input(
                f"Name for {file.name}",
                value=default_name,
                key=f"name_{file.name}"
            )
            if dataset_name:
                try:
                    # Read file based on type
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    
                    # Store in session state
                    st.session_state.datasets[dataset_name] = df
                    st.session_state.data_versions[dataset_name] = [df.copy()]
                    log_audit(f"Dataset Added: {dataset_name}")
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")

# ============ MAIN INTERFACE ============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Data Explorer", 
    "🛠 Data Ops", 
    "🔍 SQL Studio", 
    "📜 Audit Trail",
    "🚀 Deployment"
])

with tab1:  # Data Explorer
    st.subheader("🌐 Multi-Dataset Explorer")
    
    if st.session_state.datasets:
        selected_ds = st.selectbox("Choose Dataset", list(st.session_state.datasets.keys()))
        df = st.session_state.datasets[selected_ds]
        
        # Advanced DataFrame Profiling
        cols = st.columns(4)
        cols[0].metric("📦 Size", f"{df.memory_usage().sum()/1e6:.2f} MB")
        cols[1].metric("🆔 Checksum", hashlib.md5(df.to_json().encode()).hexdigest()[:8])
        cols[2].metric("⏳ Versions", len(st.session_state.data_versions.get(selected_ds, [])))
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
            if st.session_state.data_versions.get(selected_ds):
                version_compare = st.selectbox(
                    "Compare with Version", 
                    range(len(st.session_state.data_versions[selected_ds]))
                )
                st.write(f"Version {version_compare} Preview")
                st.dataframe(st.session_state.data_versions[selected_ds][version_compare].head())
            else:
                st.warning("No previous versions available")
    else:
        st.warning("No datasets uploaded yet!")

with tab2:  # Data Ops
    st.subheader("🛠 Advanced Data Operations")
    
    # Multi-Dataset Joins
    with st.expander("🔗 Data Fusion (Join N Datasets)"):
        # Join sequence management
        col1, col2 = st.columns([3,1])
        with col1:
            new_dataset = st.selectbox(
                "Add Dataset to Join", 
                list(st.session_state.datasets.keys())
            )
        with col2:
            st.write("##")
            if st.button("➕ Add to Join Sequence"):
                if new_dataset not in [ds['name'] for ds in st.session_state.join_sequence]:
                    st.session_state.join_sequence.append({
                        'name': new_dataset,
                        'join_type': 'inner',
                        'key': None
                    })
        
        # Display and configure join sequence
        if st.session_state.join_sequence:
            st.write("### Join Sequence Configuration")
            final_name = "_X_".join([ds['name'] for ds in st.session_state.join_sequence])
            
            for i, ds in enumerate(st.session_state.join_sequence):
                cols = st.columns([2,2,4,1])
                with cols[0]:
                    st.write(f"**Dataset {i+1}:** {ds['name']}")
                if i > 0:
                    with cols[1]:
                        ds['join_type'] = st.selectbox(
                            f"Join Type {i+1}",
                            ["inner", "left", "right", "outer"],
                            key=f"jt_{i}"
                        )
                    with cols[2]:
                        available_keys = get_common_columns(
                            st.session_state.datasets[st.session_state.join_sequence[i-1]['name']],
                            st.session_state.datasets[ds['name']]
                        )
                        if available_keys:
                            ds['key'] = st.selectbox(
                                f"Join Key {i+1}",
                                available_keys,
                                key=f"jk_{i}"
                            )
                        else:
                            st.error("No common columns between datasets!")
                
            if st.button("🔗 Execute Multi-Join"):
                try:
                    # Start with first dataset
                    merged = st.session_state.datasets[st.session_state.join_sequence[0]['name']].copy()
                    
                    # Iteratively join subsequent datasets
                    for i in range(1, len(st.session_state.join_sequence)):
                        current_ds = st.session_state.join_sequence[i]
                        right_df = st.session_state.datasets[current_ds['name']]
                        
                        merged = pd.merge(
                            left=merged,
                            right=right_df,
                            how=current_ds['join_type'],
                            on=current_ds['key']
                        )
                    
                    # Save final merged dataset
                    st.session_state.datasets[final_name] = merged
                    st.session_state.data_versions[final_name] = [merged.copy()]
                    log_audit(f"Merged {len(st.session_state.join_sequence)} datasets as {final_name}")
                    st.success(f"Successfully created merged dataset: {final_name}")
                    st.session_state.join_sequence = []
                except Exception as e:
                    st.error(f"Merge Error: {str(e)}")
            
            if st.button("🔄 Clear Join Sequence"):
                st.session_state.join_sequence = []

    # Data Version Control
    with st.expander("🕰️ Time Machine"):
        if st.session_state.datasets:
            selected_ds = st.selectbox("Dataset", list(st.session_state.datasets.keys()))
            versions = st.session_state.data_versions.get(selected_ds, [])
            
            if len(versions) > 0:
                version_numbers = list(range(len(versions)))
                selected_version = st.select_slider("Select Version", options=version_numbers)
                if st.button("Restore This Version"):
                    st.session_state.datasets[selected_ds] = versions[selected_version]
                    st.success(f"Restored {selected_ds} to version {selected_version}")
            else:
                st.warning("No versions available for this dataset")
        else:
            st.warning("No datasets available")

with tab3:  # SQL Studio
    st.subheader("🔍 Cross-Dataset SQL Studio")
    
    # SQL Editor
    query = st.text_area(
        "Write SQL Query", 
        height=200,
        help="Use dataset names as tables. Example: SELECT * FROM sales JOIN users ON sales.id = users.id"
    )
    
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
    
    # Bulk Export
    with st.expander("📤 Export All Data"):
        export_format = st.selectbox("Format", ["ZIP (CSV)", "ZIP (Excel)", "SQLite DB"])
        if st.button("📦 Package Data"):
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for name, df in st.session_state.datasets.items():
                    if export_format == "ZIP (CSV)":
                        zip_file.writestr(f"{name}.csv", df.to_csv(index=False))
                    elif export_format == "ZIP (Excel)":
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            df.to_excel(writer, sheet_name=name, index=False)
                        zip_file.writestr(f"{name}.xlsx", excel_buffer.getvalue())
                    elif export_format == "SQLite DB":
                        db_buffer = BytesIO()
                        engine = create_engine(f'sqlite:///{db_buffer}')
                        df.to_sql(name, engine, index=False, if_exists='replace')
                        zip_file.writestr("data.db", db_buffer.getvalue())
            st.download_button(
                "Download Package", 
                zip_buffer.getvalue(), 
                "data_package.zip"
            )
