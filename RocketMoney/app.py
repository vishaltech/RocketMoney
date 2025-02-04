import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# Replace pandas_profiling with ydata_profiling
from ydata_profiling import ProfileReport

import hashlib
import json
import os
import tempfile
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    r2_score,
    mean_squared_error
)

# --------------------------
# Configuration
# --------------------------
st.set_page_config(
    page_title="DataForge Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Authentication System
# --------------------------
class AuthSystem:
    def __init__(self):
        self.users_file = "users.json"
        self.session_timeout = 3600  # 1 hour

    def _hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def _load_users(self):
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file) as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}

    def _save_users(self, users):
        with open(self.users_file, "w") as f:
            json.dump(users, f)

    def authenticate(self):
        if 'auth' not in st.session_state:
            st.session_state.auth = {
                'authenticated': False,
                'username': None,
                'login_time': None
            }

        if not st.session_state.auth['authenticated']:
            st.title("🔒 DataForge Pro Login")
            users = self._load_users()

            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image("https://via.placeholder.com/150", width=100)
                with col2:
                    # If you see an error with horizontal=True, remove it or upgrade Streamlit.
                    auth_mode = st.radio("Mode", ["Login", "Register"], horizontal=True)

                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

                if st.button(f"{auth_mode}"):
                    if auth_mode == "Register":
                        if username and password:
                            if username not in users:
                                users[username] = self._hash_password(password)
                                self._save_users(users)
                                st.success("Account created! Please login")
                            else:
                                st.error("Username already exists")
                    else:
                        if username in users and users.get(username) == self._hash_password(password):
                            st.session_state.auth = {
                                'authenticated': True,
                                'username': username,
                                'login_time': datetime.now()
                            }
                            st.experimental_rerun()
                        else:
                            st.error("Invalid credentials")
            st.stop()
        else:
            # Session timeout check
            elapsed = (datetime.now() - st.session_state.auth['login_time']).total_seconds()
            if elapsed > self.session_timeout:
                st.session_state.auth['authenticated'] = False
                st.warning("Session expired, please login again")
                st.experimental_rerun()
            return True

# --------------------------
# Data Manager
# --------------------------
class DataManager:
    def __init__(self):
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
        if 'versions' not in st.session_state:
            st.session_state.versions = {}
        if 'lineage' not in st.session_state:
            st.session_state.lineage = {}

    def handle_upload(self):
        with st.sidebar.expander("📤 Data Upload", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload datasets",
                type=["csv", "xlsx", "parquet"],
                accept_multiple_files=True
            )

            for file in uploaded_files:
                try:
                    if file.name not in st.session_state.datasets:
                        df = self._read_file(file)
                        default_name = os.path.splitext(file.name)[0][:20]
                        new_name = st.text_input(
                            f"Name for {file.name}",
                            value=default_name,
                            key=f"name_{file.name}"
                        )
                        if new_name:
                            self._store_dataset(new_name, df, f"Uploaded {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")

    def _read_file(self, file):
        if file.name.endswith('.parquet'):
            return pd.read_parquet(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            return pd.read_csv(file)

    def _store_dataset(self, name, df, operation):
        st.session_state.datasets[name] = df
        st.session_state.versions[name] = [df.copy()]
        st.session_state.lineage[name] = [operation]

    def dataset_selector(self, key):
        if len(st.session_state.datasets) == 0:
            st.info("No datasets available. Please upload one first.")
            return None
        return st.selectbox(
            "Select Dataset",
            list(st.session_state.datasets.keys()),
            key=f"selector_{key}"
        )

# --------------------------
# Data Transformation Engine
# --------------------------
class DataTransformer:
    def __init__(self):
        self.operations = {
            "Filter": self._filter_data,
            "Merge": self._merge_datasets,
            "Clean": self._clean_data,
            "Custom": self._custom_transformation
        }

    def show_interface(self):
        with st.expander("🔧 Data Transformations", expanded=True):
            if len(st.session_state.datasets) == 0:
                st.info("No datasets available for transformation. Please upload a dataset.")
                return

            selected_op = st.selectbox("Select Operation", list(self.operations.keys()))
            self.operations[selected_op]()

    def _filter_data(self):
        dataset = DataManager().dataset_selector("filter")
        if not dataset:
            return
        df = st.session_state.datasets[dataset]

        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox("Column", df.columns, key="filter_col")
        with col2:
            operation = st.selectbox(
                "Condition",
                ["==", "!=", ">", "<", ">=", "<=", "contains", "is null", "not null"],
                key="filter_op"
            )

        value = None
        if operation not in ["is null", "not null"]:
            value = st.text_input("Value", key="filter_val")

        if st.button("Apply Filter"):
            try:
                filtered = self._apply_filter(df, column, operation, value)
                self._create_version(dataset, filtered, f"Filtered {column} {operation} {value}")
                st.success(f"Filter applied. New shape: {filtered.shape}")
            except Exception as e:
                st.error(f"Filter error: {str(e)}")

    def _apply_filter(self, df, column, operation, value):
        if operation == "==":
            return df[df[column] == value]
        elif operation == "!=":
            return df[df[column] != value]
        elif operation == ">":
            return df[df[column] > float(value)]
        elif operation == "<":
            return df[df[column] < float(value)]
        elif operation == ">=":
            return df[df[column] >= float(value)]
        elif operation == "<=":
            return df[df[column] <= float(value)]
        elif operation == "contains":
            return df[df[column].astype(str).str.contains(str(value), na=False)]
        elif operation == "is null":
            return df[df[column].isna()]
        elif operation == "not null":
            return df[df[column].notna()]

    def _merge_datasets(self):
        dm = DataManager()

        col1, col2 = st.columns(2)
        with col1:
            left_ds = dm.dataset_selector("merge_left")
            if not left_ds:
                return
            left_df = st.session_state.datasets[left_ds]
            left_key = st.selectbox("Left Key", left_df.columns)
        with col2:
            right_ds = dm.dataset_selector("merge_right")
            if not right_ds:
                return
            right_df = st.session_state.datasets[right_ds]
            right_key = st.selectbox("Right Key", right_df.columns)

        how = st.selectbox("Merge Type", ["inner", "left", "right", "outer"])
        new_name = st.text_input("New Dataset Name")

        if st.button("Merge Datasets"):
            if not new_name:
                st.error("Please enter a new dataset name.")
                return
            try:
                merged = pd.merge(
                    left_df, right_df,
                    left_on=left_key,
                    right_on=right_key,
                    how=how
                )
                self._create_version(new_name, merged, f"Merged {left_ds} and {right_ds}")
                st.success(f"Merged successfully. New shape: {merged.shape}")
            except Exception as e:
                st.error(f"Merge failed: {str(e)}")

    def _clean_data(self):
        dataset = DataManager().dataset_selector("clean")
        if not dataset:
            return
        df = st.session_state.datasets[dataset]

        st.write("### Missing Values Handling")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            col1, col2 = st.columns(2)
            with col1:
                strategy = st.selectbox("Treatment Strategy", [
                    "Drop rows",
                    "Fill with mean",
                    "Fill with median",
                    "Fill with mode",
                    "Custom value"
                ])
            with col2:
                fill_value = None
                if strategy == "Custom value":
                    fill_value = st.text_input("Custom Value")

            if st.button("Clean Data"):
                cleaned = self._handle_missing(df, strategy, fill_value)
                self._create_version(dataset, cleaned, f"Cleaned using {strategy}")
                st.success(f"Cleaning complete. New shape: {cleaned.shape}")
        else:
            st.info("No missing values found")

    def _handle_missing(self, df, strategy, custom_value=None):
        if strategy == "Drop rows":
            return df.dropna()
        elif strategy == "Fill with mean":
            return df.fillna(df.mean(numeric_only=True))
        elif strategy == "Fill with median":
            return df.fillna(df.median(numeric_only=True))
        elif strategy == "Fill with mode":
            return df.fillna(df.mode().iloc[0])
        else:
            return df.fillna(custom_value)

    def _custom_transformation(self):
        dataset = DataManager().dataset_selector("custom")
        if not dataset:
            return
        df = st.session_state.datasets[dataset]

        code = st.text_area(
            "Enter Python Code (use 'df' as DataFrame variable)",
            height=200
        )
        if st.button("Execute Code"):
            try:
                loc = {"df": df.copy(), "pd": pd, "np": np}
                exec(code, globals(), loc)
                new_df = loc.get('df', df)
                self._create_version(dataset, new_df, "Custom transformation")
                st.success("Code executed successfully")
            except Exception as e:
                st.error(f"Execution error: {str(e)}")

    def _create_version(self, name, df, operation):
        # If the dataset name doesn't already exist, initialize it properly:
        if name not in st.session_state.datasets:
            st.session_state.datasets[name] = df
            st.session_state.versions[name] = [df.copy()]
            st.session_state.lineage[name] = [operation]
        else:
            st.session_state.datasets[name] = df
            st.session_state.versions[name].append(df.copy())
            st.session_state.lineage[name].append(operation)

# --------------------------
# Visualization Engine
# --------------------------
class Visualizer:
    def show_interface(self):
        with st.expander("📊 Advanced Visualization", expanded=True):
            dataset = DataManager().dataset_selector("visualization")
            if not dataset:
                return
            df = st.session_state.datasets[dataset]

            chart_type = st.selectbox("Chart Type", [
                "Scatter Plot",
                "Line Chart",
                "Histogram",
                "Box Plot",
                "3D Scatter",
                "Heatmap"
            ])

            if chart_type == "Scatter Plot":
                self._scatter_plot(df)
            elif chart_type == "Line Chart":
                self._line_chart(df)
            elif chart_type == "Histogram":
                self._histogram(df)
            elif chart_type == "Box Plot":
                self._box_plot(df)
            elif chart_type == "3D Scatter":
                self._3d_scatter(df)
            elif chart_type == "Heatmap":
                self._heatmap(df)

    def _scatter_plot(self, df):
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("X Axis", df.columns)
        with col2:
            y = st.selectbox("Y Axis", df.columns)

        color = st.selectbox("Color", [None] + df.columns.tolist())
        size = st.selectbox("Size", [None] + df.select_dtypes(include=np.number).columns.tolist())

        fig = px.scatter(df, x=x, y=y, color=color, size=size)
        st.plotly_chart(fig, use_container_width=True)

    def _line_chart(self, df):
        x = st.selectbox("X Axis", df.columns)
        y = st.selectbox("Y Axis", df.select_dtypes(include=np.number).columns)
        color = st.selectbox("Group By", [None] + df.columns.tolist())

        fig = px.line(df, x=x, y=y, color=color)
        st.plotly_chart(fig, use_container_width=True)

    def _histogram(self, df):
        col = st.selectbox("Column", df.columns)
        nbins = st.slider("Number of Bins", 5, 100, 20)
        color = st.selectbox("Color", [None] + df.columns.tolist())

        fig = px.histogram(df, x=col, nbins=nbins, color=color)
        st.plotly_chart(fig, use_container_width=True)

    def _box_plot(self, df):
        y = st.selectbox("Value Column", df.select_dtypes(include=np.number).columns)
        x = st.selectbox("Category Column", [None] + df.columns.tolist())
        color = st.selectbox("Color", [None] + df.columns.tolist())

        fig = px.box(df, x=x, y=y, color=color)
        st.plotly_chart(fig, use_container_width=True)

    def _3d_scatter(self, df):
        cols = st.columns(3)
        with cols[0]:
            x = st.selectbox("X", df.columns)
        with cols[1]:
            y = st.selectbox("Y", df.columns)
        with cols[2]:
            z = st.selectbox("Z", df.columns)

        color = st.selectbox("Color", [None] + df.columns.tolist())
        size = st.selectbox("Size", [None] + df.select_dtypes(include=np.number).columns.tolist())

        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, size=size)
        st.plotly_chart(fig, use_container_width=True)

    def _heatmap(self, df):
        cols = st.multiselect("Select Columns", df.select_dtypes(include=np.number).columns)
        if cols:
            corr = df[cols].corr()
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Machine Learning Module
# --------------------------
class MLProcessor:
    def show_interface(self):
        with st.expander("🤖 Machine Learning Studio", expanded=True):
            dataset = DataManager().dataset_selector("ml")
            if not dataset:
                return
            df = st.session_state.datasets[dataset]

            st.write("### Dataset Preview")
            st.dataframe(df.head(3))

            target = st.selectbox("Target Variable", df.columns)
            task = st.selectbox("Task Type", ["Classification", "Regression"])
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

            if task == "Classification":
                model = RandomForestClassifier()
                metrics = {
                    'accuracy': accuracy_score,
                    'precision': precision_score,
                    'recall': recall_score
                }
            else:
                model = RandomForestRegressor()
                metrics = {
                    'r2': r2_score,
                    'mse': mean_squared_error
                }

            if st.button("Train Model"):
                try:
                    X = df.drop(columns=[target])
                    y = df[target]

                    # Convert non-numerical columns if necessary (quick fix)
                    for col in X.select_dtypes(include=['object', 'category']).columns:
                        X[col], _ = X[col].factorize()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.write("### Model Performance")
                    cols = st.columns(len(metrics))
                    for i, (name, fn) in enumerate(metrics.items()):
                        if name in ["precision", "recall"]:
                            val = fn(y_test, y_pred, average='macro', zero_division=0)
                        else:
                            val = fn(y_test, y_pred)
                        cols[i].metric(name.title(), f"{val:.2f}")

                    st.write("### Feature Importance")
                    self._show_feature_importance(model, X.columns)

                    self._save_model(model, f"{task}_{target}")
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

    def _show_feature_importance(self, model, features):
        if not hasattr(model, "feature_importances_"):
            st.info("This model does not provide feature importances.")
            return

        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

    def _save_model(self, model, name):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            joblib.dump(model, tmp.name)
            st.download_button(
                "💾 Download Model",
                data=open(tmp.name, "rb").read(),
                file_name=f"{name}_{datetime.now().strftime('%Y%m%d')}.joblib"
            )

# --------------------------
# Main Application
# --------------------------
def main():
    auth = AuthSystem()
    if auth.authenticate():
        st.title(f"🧠 DataForge Pro - Welcome {st.session_state.auth['username']}")

        data_manager = DataManager()
        data_manager.handle_upload()

        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Transformation",
            "Visualization",
            "Machine Learning",
            "Data Profiling"
        ])

        with tab1:
            DataTransformer().show_interface()

        with tab2:
            Visualizer().show_interface()

        with tab3:
            MLProcessor().show_interface()

        with tab4:
            dataset = data_manager.dataset_selector("profile")
            if dataset and st.button("Generate Profile Report"):
                # Use ydata_profiling's ProfileReport
                pr = ProfileReport(st.session_state.datasets[dataset])
                st.components.v1.html(pr.to_html(), height=800, scrolling=True)

if __name__ == "__main__":
    main()
