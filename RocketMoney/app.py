import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ydata_profiling import ProfileReport
import os
import json
import hashlib
import tempfile
import joblib
from datetime import datetime
from io import BytesIO
from deepdiff import DeepDiff

# Cloud SDKs
import boto3
from snowflake.connector import connect
from azure.storage.blob import BlobServiceClient
from google.cloud import storage

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# --------------------------
# Configuration
# --------------------------
st.set_page_config(page_title="DataForge Enterprise", layout="wide", page_icon="🚀")

# --------------------------
# Authentication System
# --------------------------
class AuthSystem:
    def __init__(self):
        self.users_file = "users.json"
        
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def load_users(self):
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file) as f:
                    return json.load(f)
        except Exception:
            return {}
    
    def save_users(self, users):
        with open(self.users_file, "w") as f:
            json.dump(users, f)
    
    def login(self):
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            
        if not st.session_state.authenticated:
            st.title("🔒 Enterprise Login")
            users = self.load_users()
            
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
            with col2:
                if st.button("Login"):
                    if users.get(username) == self.hash_password(password):
                        st.session_state.authenticated = True
                        st.session_state.user = username
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                if st.button("Register"):
                    if username and password:
                        users[username] = self.hash_password(password)
                        self.save_users(users)
                        st.success("Account created! Please login")
            st.stop()

# --------------------------
# Cloud Connection Manager
# --------------------------
class CloudManager:
    @staticmethod
    def get_connection(provider):
        secrets = st.secrets
        if provider == "aws":
            return boto3.client(
                's3',
                aws_access_key_id=secrets.aws.access_key,
                aws_secret_access_key=secrets.aws.secret_key
            )
        elif provider == "snowflake":
            return connect(
                user=secrets.snowflake.user,
                password=secrets.snowflake.password,
                account=secrets.snowflake.account,
                warehouse=secrets.snowflake.warehouse,
                database=secrets.snowflake.database
            )
        elif provider == "azure":
            return BlobServiceClient.from_connection_string(
                secrets.azure.connection_string
            )
        elif provider == "gcp":
            return storage.Client.from_service_account_info(
                json.loads(secrets.gcp.service_account)
            )

# --------------------------
# Data Operations
# --------------------------
class DataHandler:
    def __init__(self):
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
        if 'versions' not in st.session_state:
            st.session_state.versions = {}
        if 'lineage' not in st.session_state:
            st.session_state.lineage = {}
    
    def import_data(self, provider, config):
        try:
            if provider == "aws":
                return self._import_aws(config)
            elif provider == "snowflake":
                return self._import_snowflake(config)
            elif provider == "azure":
                return self._import_azure(config)
            elif provider == "gcp":
                return self._import_gcp(config)
        except Exception as e:
            st.error(f"Import failed: {str(e)}")
    
    def _import_aws(self, config):
        s3 = CloudManager.get_connection("aws")
        response = s3.list_objects_v2(Bucket=config['bucket'], Prefix=config['prefix'])
        files = [obj['Key'] for obj in response.get('Contents', [])]
        selected = st.selectbox("Select S3 Object", files)
        
        with tempfile.NamedTemporaryFile() as tmp:
            s3.download_file(config['bucket'], selected, tmp.name)
            df = pd.read_parquet(tmp) if selected.endswith('.parquet') else pd.read_csv(tmp)
            return df
    
    def _import_snowflake(self, config):
        conn = CloudManager.get_connection("snowflake")
        query = config['query'] if config['query'] else f"SELECT * FROM {config['table']}"
        return pd.read_sql(query, conn)
    
    def _import_azure(self, config):
        client = CloudManager.get_connection("azure")
        container = client.get_container_client(config['container'])
        blobs = [blob.name for blob in container.list_blobs(name_starts_with=config['prefix'])]
        selected = st.selectbox("Select Azure Blob", blobs)
        
        blob_client = container.get_blob_client(selected)
        data = blob_client.download_blob().readall()
        return pd.read_parquet(BytesIO(data)) if selected.endswith('.parquet') else pd.read_csv(BytesIO(data))
    
    def _import_gcp(self, config):
        client = CloudManager.get_connection("gcp")
        bucket = client.get_bucket(config['bucket'])
        blobs = list(bucket.list_blobs(prefix=config['prefix']))
        selected = st.selectbox("Select GCS Object", [blob.name for blob in blobs])
        
        blob = bucket.blob(selected)
        with tempfile.NamedTemporaryFile() as tmp:
            blob.download_to_filename(tmp.name)
            return pd.read_parquet(tmp) if selected.endswith('.parquet') else pd.read_csv(tmp)

# --------------------------
# ML Pipeline
# --------------------------
class MLProcessor:
    def train_model(self, df, target, task):
        X = df.drop(columns=[target])
        y = df[target]
        
        if task == "classification":
            model = RandomForestClassifier()
            metric = accuracy_score
        else:
            model = RandomForestRegressor()
            metric = r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = metric(y_test, preds)
        
        return model, score

# --------------------------
# Main Application
# --------------------------
class DataForgeApp:
    def __init__(self):
        self.auth = AuthSystem()
        self.data_handler = DataHandler()
        self.ml = MLProcessor()
        
    def run(self):
        self.auth.login()
        self._setup_sidebar()
        self._main_interface()
    
    def _setup_sidebar(self):
        with st.sidebar:
            st.header("☁️ Cloud Connections")
            self.provider = st.selectbox("Select Provider", ["AWS", "Snowflake", "Azure", "GCP"])
            
            with st.expander("Configuration"):
                self.cloud_config = self._get_provider_config()
            
            if st.button("Import Data"):
                df = self.data_handler.import_data(self.provider.lower(), self.cloud_config)
                if df is not None:
                    dataset_name = st.text_input("Dataset Name", value=f"{self.provider}_import")
                    st.session_state.datasets[dataset_name] = df
                    st.session_state.versions[dataset_name] = [df.copy()]
                    st.session_state.lineage[dataset_name] = [f"Imported from {self.provider}"]
    
    def _get_provider_config(self):
        config = {}
        if self.provider == "AWS":
            config['bucket'] = st.text_input("S3 Bucket")
            config['prefix'] = st.text_input("S3 Prefix")
        elif self.provider == "Snowflake":
            config['table'] = st.text_input("Table Name")
            config['query'] = st.text_area("Custom Query")
        elif self.provider == "Azure":
            config['container'] = st.text_input("Container Name")
            config['prefix'] = st.text_input("Blob Prefix")
        elif self.provider == "GCP":
            config['bucket'] = st.text_input("GCS Bucket")
            config['prefix'] = st.text_input("Object Prefix")
        return config
    
    def _main_interface(self):
        tabs = st.tabs(["Data Explorer", "ML Studio", "Lineage", "Export"])
        
        with tabs[0]:
            self._data_explorer()
        
        with tabs[1]:
            self._ml_studio()
        
        with tabs[2]:
            self._show_lineage()
        
        with tabs[3]:
            self._export_data()
    
    def _data_explorer(self):
        st.header("🔍 Data Explorer")
        if st.session_state.datasets:
            dataset = st.selectbox("Select Dataset", list(st.session_state.datasets.keys()))
            df = st.session_state.datasets[dataset]
            
            cols = st.columns(4)
            cols[0].metric("Rows", df.shape[0])
            cols[1].metric("Columns", df.shape[1])
            cols[2].metric("Memory", f"{df.memory_usage().sum()/1e6:.2f} MB")
            cols[3].metric("Versions", len(st.session_state.versions[dataset]))
            
            with st.expander("Data Profile"):
                if st.button("Generate Report"):
                    pr = ProfileReport(df, title=f"{dataset} Profile")
                    st.components.v1.html(pr.to_html(), height=800, scrolling=True)
            
            with st.expander("Data Visualization"):
                self._visualize_data(df)
        else:
            st.info("Import data from the sidebar")
    
    def _visualize_data(self, df):
        x_axis = st.selectbox("X Axis", df.columns)
        y_axis = st.selectbox("Y Axis", df.columns)
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Histogram"])
        
        if chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis)
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _ml_studio(self):
        st.header("🤖 ML Studio")
        if st.session_state.datasets:
            dataset = st.selectbox("Select Dataset", list(st.session_state.datasets.keys()))
            df = st.session_state.datasets[dataset]
            
            target = st.selectbox("Target Variable", df.columns)
            task = st.selectbox("Task Type", ["Classification", "Regression"])
            
            if st.button("Train Model"):
                model, score = self.ml.train_model(df, target, task.lower())
                st.session_state.model = model
                st.success(f"Model trained with score: {score:.2f}")
                
                with st.expander("Model Details"):
                    st.write(model.get_params())
                    
                    if hasattr(model, "feature_importances_"):
                        features = df.drop(columns=[target]).columns
                        importance = pd.DataFrame({
                            "Feature": features,
                            "Importance": model.feature_importances_
                        }).sort_values("Importance", ascending=False)
                        fig = px.bar(importance, x="Importance", y="Feature")
                        st.plotly_chart(fig)
        else:
            st.info("Import data first")
    
    def _show_lineage(self):
        st.header("🔗 Data Lineage")
        if st.session_state.datasets:
            dataset = st.selectbox("Dataset", list(st.session_state.datasets.keys()))
            lineage = st.session_state.lineage[dataset]
            
            dot = Digraph()
            for i, step in enumerate(lineage):
                dot.node(str(i), f"v{i}: {step}")
                if i > 0:
                    dot.edge(str(i-1), str(i))
            st.graphviz_chart(dot)
    
    def _export_data(self):
        st.header("📤 Data Export")
        if st.session_state.datasets:
            dataset = st.selectbox("Dataset", list(st.session_state.datasets.keys()))
            export_format = st.selectbox("Format", ["CSV", "Parquet"])
            provider = st.selectbox("Destination", ["AWS", "Azure", "GCP"])
            
            if st.button("Export"):
                df = st.session_state.datasets[dataset]
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{dataset}_{timestamp}.{export_format.lower()}"
                
                with tempfile.NamedTemporaryFile() as tmp:
                    if export_format == "CSV":
                        df.to_csv(tmp.name, index=False)
                    else:
                        df.to_parquet(tmp.name)
                    
                    if provider == "AWS":
                        s3 = CloudManager.get_connection("aws")
                        s3.upload_file(tmp.name, self.cloud_config['bucket'], f"{self.cloud_config['prefix']}/{filename}")
                    elif provider == "Azure":
                        client = CloudManager.get_connection("azure")
                        blob = client.get_blob_client(self.cloud_config['container'], f"{self.cloud_config['prefix']}/{filename}")
                        blob.upload_blob(tmp.read())
                    elif provider == "GCP":
                        client = CloudManager.get_connection("gcp")
                        bucket = client.get_bucket(self.cloud_config['bucket'])
                        blob = bucket.blob(f"{self.cloud_config['prefix']}/{filename}")
                        blob.upload_from_filename(tmp.name)
                    
                    st.success(f"Exported {filename} to {provider}")

if __name__ == "__main__":
    app = DataForgeApp()
    app.run()
