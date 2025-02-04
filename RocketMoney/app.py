import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas_profiling import ProfileReport
import os
import json
from datetime import datetime
import tempfile

# Cloud Provider Imports
import boto3  # AWS
from snowflake import connector  # Snowflake
from azure.storage.blob import BlobServiceClient  # Azure
from google.cloud import storage  # GCP

# -----------------------
# Configuration Setup
# -----------------------
st.set_page_config(page_title="Cloud DataForge", layout="wide", page_icon="🌩️")

# -----------------------
# Cloud Connection Managers
# -----------------------
class CloudConnector:
    @staticmethod
    def aws_client():
        return boto3.client(
            's3',
            aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY"),
            aws_secret_access_key=st.secrets.get("AWS_SECRET_KEY")
        )

    @staticmethod
    def snowflake_conn():
        return connector.connect(
            user=st.secrets.get("SNOWFLAKE_USER"),
            password=st.secrets.get("SNOWFLAKE_PWD"),
            account=st.secrets.get("SNOWFLAKE_ACCOUNT"),
            warehouse=st.secrets.get("SNOWFLAKE_WH"),
            database=st.secrets.get("SNOWFLAKE_DB")
        )

    @staticmethod
    def azure_client():
        return BlobServiceClient.from_connection_string(
            st.secrets.get("AZURE_CONN_STR")
        )

    @staticmethod
    def gcp_client():
        return storage.Client.from_service_account_json(
            json.loads(st.secrets.get("GCP_CREDENTIALS"))
        )

# -----------------------
# Core Application
# -----------------------
def main():
    st.title("🌩️ Multi-Cloud Data Platform")
    
    # Initialize session state
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    
    # -----------------------
    # Cloud Connection Sidebar
    # -----------------------
    with st.sidebar:
        st.header("🔗 Cloud Connections")
        with st.expander("AWS Configuration"):
            aws_bucket = st.text_input("S3 Bucket")
            aws_prefix = st.text_input("S3 Path Prefix")
            
        with st.expander("Snowflake Configuration"):
            sf_table = st.text_input("Table Name")
            sf_query = st.text_area("Custom Query")
            
        with st.expander("Azure Configuration"):
            az_container = st.text_input("Container Name")
            az_prefix = st.text_input("Blob Prefix")
            
        with st.expander("GCP Configuration"):
            gcp_bucket = st.text_input("GCS Bucket")
            gcp_prefix = st.text_input("GCS Path Prefix")

    # -----------------------
    # Cloud Data Operations
    # -----------------------
    def handle_aws_import():
        s3 = CloudConnector.aws_client()
        response = s3.list_objects_v2(Bucket=aws_bucket, Prefix=aws_prefix)
        files = [obj['Key'] for obj in response.get('Contents', [])]
        
        selected = st.selectbox("Select S3 Object", files)
        if st.button("Import from S3"):
            with tempfile.NamedTemporaryFile() as tmp:
                s3.download_file(aws_bucket, selected, tmp.name)
                df = pd.read_parquet(tmp) if selected.endswith('.parquet') else pd.read_csv(tmp)
                st.session_state.datasets[selected.split('/')[-1]] = df

    def handle_snowflake_import():
        conn = CloudConnector.snowflake_conn()
        cursor = conn.cursor()
        
        if sf_query:
            cursor.execute(sf_query)
        else:
            cursor.execute(f"SELECT * FROM {sf_table}")
            
        df = cursor.fetch_pandas_all()
        st.session_state.datasets[sf_table] = df

    def handle_azure_import():
        client = CloudConnector.azure_client()
        container = client.get_container_client(az_container)
        blobs = [blob.name for blob in container.list_blobs(name_starts_with=az_prefix)]
        
        selected = st.selectbox("Select Azure Blob", blobs)
        if st.button("Import from Azure"):
            blob = container.get_blob_client(selected)
            data = blob.download_blob().readall()
            df = pd.read_parquet(BytesIO(data)) if selected.endswith('.parquet') else pd.read_csv(BytesIO(data))
            st.session_state.datasets[selected.split('/')[-1]] = df

    def handle_gcp_import():
        client = CloudConnector.gcp_client()
        bucket = client.get_bucket(gcp_bucket)
        blobs = list(bucket.list_blobs(prefix=gcp_prefix))
        
        selected = st.selectbox("Select GCS Object", [blob.name for blob in blobs])
        if st.button("Import from GCS"):
            blob = bucket.blob(selected)
            with tempfile.NamedTemporaryFile() as tmp:
                blob.download_to_filename(tmp.name)
                df = pd.read_parquet(tmp) if selected.endswith('.parquet') else pd.read_csv(tmp)
                st.session_state.datasets[selected.split('/')[-1]] = df

    # -----------------------
    # Main Interface
    # -----------------------
    tabs = st.tabs(["Cloud Import", "Data Explorer", "Profiling", "Export"])
    
    with tabs[0]:
        cloud_provider = st.radio("Select Cloud Provider", 
                                ["AWS", "Snowflake", "Azure", "GCP"])
        
        if cloud_provider == "AWS":
            handle_aws_import()
        elif cloud_provider == "Snowflake":
            handle_snowflake_import()
        elif cloud_provider == "Azure":
            handle_azure_import()
        elif cloud_provider == "GCP":
            handle_gcp_import()
    
    with tabs[1]:
        if st.session_state.datasets:
            ds = st.selectbox("Select Dataset", list(st.session_state.datasets.keys()))
            st.dataframe(st.session_state.datasets[ds].head())
        else:
            st.info("Import data from cloud providers")
    
    with tabs[2]:
        if st.session_state.datasets:
            ds = st.selectbox("Choose Dataset", list(st.session_state.datasets.keys()))
            if st.button("Generate Profile"):
                pr = ProfileReport(st.session_state.datasets[ds])
                st.components.v1.html(pr.to_html(), height=800, scrolling=True)
    
    with tabs[3]:
        if st.session_state.datasets:
            ds = st.selectbox("Select Dataset to Export", list(st.session_state.datasets.keys()))
            export_format = st.selectbox("Format", ["CSV", "Parquet"])
            export_provider = st.selectbox("Destination", ["AWS", "Azure", "GCP"])
            
            if st.button(f"Export to {export_provider}"):
                df = st.session_state.datasets[ds]
                with tempfile.NamedTemporaryFile() as tmp:
                    if export_format == "CSV":
                        df.to_csv(tmp.name, index=False)
                    else:
                        df.to_parquet(tmp.name)
                    
                    timestamp = datetime.now().strftime("%Y%m%d%H%M")
                    filename = f"{ds}_{timestamp}.{export_format.lower()}"
                    
                    if export_provider == "AWS":
                        CloudConnector.aws_client().upload_file(tmp.name, aws_bucket, f"{aws_prefix}/{filename}")
                    elif export_provider == "Azure":
                        client = CloudConnector.azure_client()
                        blob = client.get_blob_client(az_container, f"{az_prefix}/{filename}")
                        with open(tmp.name, "rb") as data:
                            blob.upload_blob(data)
                    elif export_provider == "GCP":
                        bucket = CloudConnector.gcp_client().get_bucket(gcp_bucket)
                        blob = bucket.blob(f"{gcp_prefix}/{filename}")
                        blob.upload_from_filename(tmp.name)
                    
                    st.success(f"Exported {filename} to {export_provider}")

if __name__ == "__main__":
    main()
