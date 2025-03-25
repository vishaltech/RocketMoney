import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Try to import st_aggrid; if unavailable, fallback to st.dataframe.
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    use_aggrid = True
except ImportError:
    use_aggrid = False

from streamlit_ace import st_ace
from pandasql import sqldf

# Set page configuration and custom styling
st.set_page_config(page_title="Advanced Data Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #f9fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to process uploaded files
def process_file(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    try:
        if ext == ".csv":
            return {"Sheet1": pd.read_csv(uploaded_file)}
        elif ext == ".xlsx":
            # Read all sheets from Excel file
            return pd.read_excel(uploaded_file, sheet_name=None)
        elif ext == ".parquet":
            return {"Sheet1": pd.read_parquet(uploaded_file)}
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Sidebar: File uploader and sheet selector
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV, Excel, or Parquet file", type=["csv", "xlsx", "parquet"])
if uploaded_file:
    data = process_file(uploaded_file)
    if data is None:
        st.stop()
    sheet_names = list(data.keys())
    sheet_selected = st.sidebar.selectbox("Select Sheet", sheet_names)
    df = data[sheet_selected]
else:
    st.info("Please upload a data file to begin.")
    st.stop()

# Create tabs for functionalities
(tab_explorer, tab_summary, tab_visualization, tab_sql, tab_tools, tab_analytics) = st.tabs([
    "Data Explorer", "Data Summary", "Visualization", "SQL Query", "Data Tools", "Advanced Analytics"
])

# --- Tab 1: Data Explorer ---
with tab_explorer:
    st.subheader("Data Explorer")
    st.write("Preview your dataset:")
    if use_aggrid:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=True)
        gb.configure_side_bar()
        grid_options = gb.build()
        AgGrid(df, gridOptions=grid_options, height=500, theme="streamlit")
    else:
        st.dataframe(df)

# --- Tab 2: Data Summary ---
with tab_summary:
    st.subheader("Data Summary")
    st.write(f"**Shape:** {df.shape}")
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe(include='all').T)
    st.write("**Missing Values:**")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.dataframe(missing.to_frame(name="Missing Count"))
    else:
        st.write("No missing values detected.")

# --- Tab 3: Visualization ---
with tab_visualization:
    st.subheader("Visualization")
    # Get list of numeric and categorical columns
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if numeric_columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("Select X-Axis", numeric_columns, key="viz_x")
        with col2:
            y_axis = st.selectbox("Select Y-Axis", numeric_columns, key="viz_y")
        with col3:
            color_option = st.selectbox("Color By (optional)", [None] + categorical_columns, key="viz_color")
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Box"])
        if st.button("Generate Chart"):
            try:
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option)
                elif chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_option)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_option)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, color=color_option)
                elif chart_type == "Box":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_option)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating chart: {e}")
    else:
        st.write("Not enough numeric columns available for visualization.")

# --- Tab 4: SQL Query ---
with tab_sql:
    st.subheader("SQL Query")
    st.write("Query your data using SQL. Use `df` as the table name.")
    query = st_ace(
        language="sql",
        theme="monokai",
        key="sql_editor",
        height=200,
        placeholder="e.g., SELECT * FROM df LIMIT 10",
    )
    if st.button("Run Query"):
        try:
            result = sqldf(query, {"df": df})
            st.write("Query Result:")
            st.dataframe(result)
        except Exception as e:
            st.error(f"SQL Query Error: {e}")

# --- Tab 5: Data Tools ---
with tab_tools:
    st.subheader("Data Tools")
    st.write("Basic data cleaning options:")
    if st.button("Drop Duplicates"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicates dropped.")
    clean_option = st.selectbox("Fill Missing Values With", ["None", "Mean", "Median", "Mode", "Constant"], key="clean_opt")
    if clean_option != "None":
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype in [np.float64, np.int64]:
                if clean_option == "Mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif clean_option == "Median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif clean_option == "Mode":
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                elif clean_option == "Constant":
                    df_clean[col].fillna(0, inplace=True)
            else:
                if clean_option == "Mode":
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                elif clean_option == "Constant":
                    df_clean[col].fillna("Unknown", inplace=True)
        st.write("Cleaned Data Preview:")
        st.dataframe(df_clean.head())
        csv_data = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    else:
        st.write("No cleaning operation selected.")

# --- Tab 6: Advanced Analytics ---
with tab_analytics:
    st.subheader("Advanced Analytics")
    st.write("**Correlation Matrix** for numeric variables:")
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        corr = num_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Viridis',
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            autosize=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No numeric columns available for correlation analysis.")
