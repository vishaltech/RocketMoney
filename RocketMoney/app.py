import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import hashlib
import datetime
import os
import json
import tempfile
import zipfile
from graphviz import Digraph
from sqlalchemy import create_engine
import pyarrow as pa
import pyarrow.parquet as pq
from ydata_profiling import ProfileReport
import joblib
from deepdiff import DeepDiff

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error, 
                            classification_report, confusion_matrix)

# -----------------------
# Initial Setup
# -----------------------
st.set_page_config(page_title="🚀 DataForge Pro", layout="wide", page_icon="🔮")

# -----------------------
# Authentication System
# -----------------------
def user_manager():
    USERS_FILE = "users.json"
    
    def load_users():
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, "r") as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}

    def save_users(users):
        with open(USERS_FILE, "w") as f:
            json.dump(users, f)

    if "auth" not in st.session_state:
        st.session_state.auth = {"authenticated": False}

    if st.session_state.auth["authenticated"]:
        return True

    st.title("🔒 DataForge Pro Access")
    users = load_users()
    auth_mode = "Register" if not users else st.radio("Mode", ("Login", "Register"))

    if auth_mode == "Register":
        with st.form("Register"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            pwd2 = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register"):
                if user and pwd and (pwd == pwd2):
                    users[user] = hashlib.sha256(pwd.encode()).hexdigest()
                    save_users(users)
                    st.success("Registered! Please login")
                return False

    elif auth_mode == "Login":
        with st.form("Login"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if users.get(user) == hashlib.sha256(pwd.encode()).hexdigest():
                    st.session_state.auth = {"authenticated": True, "user": user}
                    st.rerun()
                else:
                    st.error("Invalid credentials")
                return False
    return False

if not user_manager():
    st.stop()

# -----------------------
# Core Application
# -----------------------
class DataForge:
    def __init__(self):
        self.init_session()
        
    def init_session(self):
        defaults = {
            "datasets": {},
            "versions": {},
            "lineage": {},
            "audit": [],
            "queries": [],
            "models": {}
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    def log_audit(self, action):
        entry = f"{datetime.datetime.now()} | {st.session_state.auth['user']} | {action}"
        st.session_state.audit.append(entry)

    def handle_upload(self):
        with st.sidebar:
            st.header("📤 Data Hub")
            files = st.file_uploader("Upload datasets", 
                                   type=["csv", "xlsx", "parquet", "feather"],
                                   accept_multiple_files=True)
            
            if files:
                for file in files:
                    try:
                        if file.name.endswith(('.xls', '.xlsx')):
                            sheets = pd.read_excel(file, sheet_name=None)
                            for sheet, df in sheets.items():
                                self.store_dataset(file.name, sheet, df)
                        else:
                            ext = file.name.split('.')[-1]
                            df = pd.read_csv(file) if ext == 'csv' else pd.read_parquet(file)
                            self.store_dataset(file.name, 'data', df)
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {str(e)}")

    def store_dataset(self, filename, sheet, df):
        default_name = f"{filename.split('.')[0]}_{sheet}"[:20]
        name = st.text_input(f"Name for {filename} - {sheet}", value=default_name)
        if name and name not in st.session_state.datasets:
            st.session_state.datasets[name] = df
            st.session_state.versions[name] = [df.copy()]
            st.session_state.lineage[name] = [f"Imported from {filename}"]
            self.log_audit(f"Dataset added: {name} ({len(df)} rows)")

    def data_explorer(self):
        st.header("🔍 Data Explorer")
        ds = st.selectbox("Choose dataset", list(st.session_state.datasets.keys()))
        df = st.session_state.datasets[ds]
        
        cols = st.columns(4)
        cols[0].metric("Rows", df.shape[0])
        cols[1].metric("Columns", df.shape[1])
        cols[2].metric("Memory", f"{df.memory_usage().sum()/1e6:.2f} MB")
        cols[3].metric("Versions", len(st.session_state.versions[ds]))
        
        with st.expander("🔬 Data Profile"):
            if st.button("Generate Profile"):
                pr = ProfileReport(df, title=f"Profile for {ds}")
                st.components.v1.html(pr.to_html(), height=800, scrolling=True)
        
        with st.expander("📊 Visual Analysis"):
            self.visual_analysis(df)
    
    def visual_analysis(self, df):
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("X Axis", df.columns)
        with col2:
            y = st.selectbox("Y Axis", df.columns if len(df.columns) > 1 else [None])
        
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Histogram", "Box"])
        if chart_type == "Scatter":
            fig = px.scatter(df, x=x, y=y, hover_name=df.index)
        elif chart_type == "Line":
            fig = px.line(df, x=x, y=y)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x)
        elif chart_type == "Box":
            fig = px.box(df, x=x, y=y if y else None)
        st.plotly_chart(fig, use_container_width=True)

    def ml_studio(self):
        st.header("🤖 ML Studio")
        ds = st.selectbox("Select Dataset", list(st.session_state.datasets.keys()))
        df = st.session_state.datasets[ds]
        
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        task = st.radio("Task Type", ["Classification", "Regression"])
        target = st.selectbox("Target Variable", df.columns)
        features = st.multiselect("Features", [c for c in df.columns if c != target])
        
        model_type = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "SVM", "Linear Model"])
        self.model_training(task, model_type, df, features, target)
    
    def model_training(self, task, model_type, df, features, target):
        with st.expander("⚙️ Hyperparameters"):
            params = self.get_model_params(task, model_type)
        
        if st.button("Train Model"):
            try:
                X = df[features].apply(pd.to_numeric, errors='coerce')
                y = df[target]
                
                if task == "Classification":
                    y = y.astype('category').cat.codes
                else:
                    y = pd.to_numeric(y, errors='coerce')
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = self.init_model(task, model_type, params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                self.show_metrics(task, y_test, preds)
                self.feature_analysis(model, features)
                self.save_model(model, model_type)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
    
    def get_model_params(self, task, model_type):
        params = {}
        if model_type == "Random Forest":
            params["n_estimators"] = st.slider("Trees", 10, 500, 100)
            params["max_depth"] = st.slider("Max Depth", 2, 50, 10)
        elif model_type == "Gradient Boosting":
            params["n_estimators"] = st.slider("Estimators", 10, 500, 100)
            params["learning_rate"] = st.slider("Learning Rate", 0.001, 1.0, 0.1)
        elif model_type == "SVM":
            params["C"] = st.slider("Regularization", 0.1, 10.0, 1.0)
            params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        return params
    
    def init_model(self, task, model_type, params):
        if task == "Classification":
            models = {
                "Random Forest": RandomForestClassifier,
                "Gradient Boosting": GradientBoostingClassifier,
                "SVM": SVC,
                "Linear Model": LogisticRegression
            }
        else:
            models = {
                "Random Forest": RandomForestRegressor,
                "Gradient Boosting": GradientBoostingRegressor,
                "SVM": SVR,
                "Linear Model": LinearRegression
            }
        return models[model_type](**params)
    
    def show_metrics(self, task, y_true, y_pred):
        st.subheader("📊 Performance Metrics")
        if task == "Classification":
            st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
            st.write(classification_report(y_true, y_pred))
            fig = px.imshow(confusion_matrix(y_true, y_pred), 
                          text_auto=True, labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig)
        else:
            st.metric("R² Score", f"{r2_score(y_true, y_pred):.2f}")
            st.metric("RMSE", f"{mean_squared_error(y_true, y_pred, squared=False):.2f}")
            fig = px.scatter(x=y_true, y=y_pred, labels={"x": "Actual", "y": "Predicted"})
            fig.add_shape(type="line", x0=y_true.min(), y0=y_true.min(),
                        x1=y_true.max(), y1=y_true.max())
            st.plotly_chart(fig)
    
    def feature_analysis(self, model, features):
        st.subheader("📈 Feature Analysis")
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            fig = px.bar(importance, x="Importance", y="Feature", orientation='h')
            st.plotly_chart(fig)
        elif hasattr(model, "coef_"):
            coefs = pd.DataFrame({
                "Feature": features,
                "Coefficient": model.coef_
            }).sort_values("Coefficient", ascending=False)
            fig = px.bar(coefs, x="Coefficient", y="Feature", orientation='h')
            st.plotly_chart(fig)
    
    def save_model(self, model, model_type):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            joblib.dump(model, tmp.name)
            st.download_button(
                "💾 Download Model",
                data=open(tmp.name, "rb"),
                file_name=f"{model_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.joblib"
            )

    def data_lineage(self):
        st.header("🔗 Data Lineage")
        ds = st.selectbox("Dataset", list(st.session_state.datasets.keys()))
        lineage = st.session_state.lineage[ds]
        
        dot = Digraph()
        for i, step in enumerate(lineage):
            dot.node(str(i), f"v{i}: {step}")
            if i > 0:
                dot.edge(str(i-1), str(i))
        st.graphviz_chart(dot)
        
        if st.button("View Version Diffs"):
            self.show_version_diffs(ds)
    
    def show_version_diffs(self, ds):
        versions = st.session_state.versions[ds]
        current = st.session_state.datasets[ds]
        
        for i, ver in enumerate(versions[:-1]):
            diff = DeepDiff(ver, versions[i+1], ignore_order=True)
            with st.expander(f"Diff v{i} → v{i+1}"):
                st.json(diff.to_json())

    def run(self):
        self.handle_upload()
        tabs = st.tabs(["Explorer", "ML Studio", "Lineage", "Audit"])
        
        with tabs[0]:
            self.data_explorer()
        
        with tabs[1]:
            self.ml_studio()
        
        with tabs[2]:
            self.data_lineage()
        
        with tabs[3]:
            st.header("📜 Audit Trail")
            st.write(f"Total entries: {len(st.session_state.audit)}")
            for entry in reversed(st.session_state.audit[-20:]):
                st.code(entry)

# -----------------------
# Run Application
# -----------------------
if __name__ == "__main__":
    app = DataForge()
    app.run()
