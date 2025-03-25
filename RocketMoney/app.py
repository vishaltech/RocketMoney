import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import graphviz
from streamlit_ace import st_ace
from pandasql import sqldf

# Optional interactive grid: if not installed, fallback to st.dataframe.
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    use_aggrid = True
except ImportError:
    use_aggrid = False

# Optional: Google Drive integration using PyDrive2.
try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    use_gdrive = True
except ImportError:
    use_gdrive = False

# -------------------------------------------------------------------
# Page Configuration & Custom Styling
# -------------------------------------------------------------------
st.set_page_config(page_title="Ultimate Data Analyzer Pro", layout="wide")
st.markdown("""
<style>
  body { background: #f4f7f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
  .sidebar .sidebar-content { background: #ffffff; }
  /* Card styling for dashboard metrics */
  .card {
      background-color: #ffffff;
      padding: 20px;
      border-radius: 12px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      margin: 10px;
  }
  .card h3 { margin: 0 0 10px 0; color: #333; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def process_file(uploaded_file):
    """Load a CSV, Excel, or Parquet file into a dict of DataFrames."""
    ext = Path(uploaded_file.name).suffix.lower()
    try:
        if ext == ".csv":
            return {"Sheet1": pd.read_csv(uploaded_file)}
        elif ext == ".xlsx":
            return pd.read_excel(uploaded_file, sheet_name=None)
        elif ext == ".parquet":
            return {"Sheet1": pd.read_parquet(uploaded_file)}
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Google Drive helper functions (requires PyDrive2 and proper credentials)
def authenticate_drive():
    """Authenticate with Google Drive via OAuth. Ensure 'client_secrets.json' is provided."""
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # This will open a local webserver for OAuth
    drive = GoogleDrive(gauth)
    return drive

def list_drive_files(drive, folder_id="root"):
    """List files in a specified Google Drive folder."""
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    return file_list

def upload_file_to_drive(drive, file_path, file_title):
    """Upload a local file to Google Drive."""
    file_drive = drive.CreateFile({'title': file_title})
    file_drive.SetContentFile(file_path)
    file_drive.Upload()
    return file_drive['id']

# -------------------------------------------------------------------
# Sidebar: Upload Primary & Secondary Data
# -------------------------------------------------------------------
st.sidebar.header("Upload Primary Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV, Excel, or Parquet", type=["csv", "xlsx", "parquet"])
if uploaded_file:
    data = process_file(uploaded_file)
    if data is None:
        st.stop()
    sheet_names = list(data.keys())
    sheet_selected = st.sidebar.selectbox("Select Sheet", sheet_names)
    df = data[sheet_selected]
else:
    st.info("Please upload a primary data file.")
    st.stop()

st.sidebar.header("Upload Secondary Data (Optional)")
uploaded_file2 = st.sidebar.file_uploader("Secondary File", type=["csv", "xlsx", "parquet"], key="file2")
if uploaded_file2:
    data2 = process_file(uploaded_file2)
    if data2:
        sheet_names2 = list(data2.keys())
        sheet_selected2 = st.sidebar.selectbox("Select Sheet (Secondary)", sheet_names2, key="sheet2")
        df2 = data2[sheet_selected2]
    else:
        df2 = None
else:
    df2 = None

# -------------------------------------------------------------------
# Initialize Session State for Data Lineage & SQL History
# -------------------------------------------------------------------
if "lineage_steps" not in st.session_state:
    st.session_state.lineage_steps = []
if "sql_history" not in st.session_state:
    st.session_state.sql_history = []

# -------------------------------------------------------------------
# Create Tabs for All Features (10 Tabs including Google Drive)
# -------------------------------------------------------------------
tabs = st.tabs([
    "Data Preview & Filtering",
    "Summary & Profiling",
    "Advanced Visualizations",
    "SQL Query Editor",
    "Data Lineage & Transformations",
    "Pivot Table Builder",
    "Advanced Analytics",
    "Custom Dashboard",
    "Data Merge",
    "Google Drive Integration"
])

# -------------------------------------------------------------------
# Tab 1: Data Preview & Advanced Filtering
# -------------------------------------------------------------------
with tabs[0]:
    st.header("Data Preview & Filtering")
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to display", all_columns, default=all_columns)
    df_display = df[selected_columns] if selected_columns else df
    with st.expander("Add Filters"):
        filter_conditions = []
        for col in df_display.columns:
            if pd.api.types.is_numeric_dtype(df_display[col]):
                min_val, max_val = float(df_display[col].min()), float(df_display[col].max())
                condition = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
                filter_conditions.append((col, condition))
            elif pd.api.types.is_string_dtype(df_display[col]):
                unique_vals = df_display[col].dropna().unique().tolist()
                if unique_vals:
                    condition = st.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
                    filter_conditions.append((col, condition))
        for col, condition in filter_conditions:
            if isinstance(condition, tuple):
                df_display = df_display[(df_display[col] >= condition[0]) & (df_display[col] <= condition[1])]
            elif isinstance(condition, list):
                df_display = df_display[df_display[col].isin(condition)]
    search_query = st.text_input("Search (in displayed columns)")
    if search_query:
        mask = df_display.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)
        df_display = df_display[mask]
    if use_aggrid:
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=True)
        gb.configure_side_bar()
        grid_options = gb.build()
        AgGrid(df_display, gridOptions=grid_options, height=500, theme="streamlit")
    else:
        st.dataframe(df_display)

# -------------------------------------------------------------------
# Tab 2: Summary & Profiling
# -------------------------------------------------------------------
with tabs[1]:
    st.header("Summary & Profiling")
    st.write("**Data Shape:**", df.shape)
    st.write("**Data Types:**")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Type"]))
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe(include='all').T)
    st.write("**Missing Values:**")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        fig_missing = px.bar(x=missing.index, y=missing.values, labels={'x':"Column", 'y':"Missing Count"},
                              title="Missing Values Count")
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.write("No missing values detected.")
    st.subheader("Outlier Detection (IQR Method)")
    outlier_info = {}
    for col in df.select_dtypes(include=np.number).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = outliers.shape[0]
    st.dataframe(pd.DataFrame(outlier_info, index=["Outlier Count"]).T)

# -------------------------------------------------------------------
# Tab 3: Advanced Visualizations
# -------------------------------------------------------------------
with tabs[2]:
    st.header("Advanced Visualizations")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if numeric_columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X-Axis", numeric_columns, key="viz_x")
        with col2:
            y_axis = st.selectbox("Y-Axis", numeric_columns, key="viz_y")
        with col3:
            color_option = st.selectbox("Color By (optional)", [None] + categorical_columns, key="viz_color")
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin"], key="chart_type")
        facet_row = st.selectbox("Facet Row (optional)", [None] + categorical_columns, key="facet_row")
        facet_col = st.selectbox("Facet Column (optional)", [None] + categorical_columns, key="facet_col")
        if st.button("Generate Chart", key="gen_chart"):
            try:
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option,
                                     facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_option,
                                  facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_option,
                                 facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, color=color_option,
                                       facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Box":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_option,
                                 facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Violin":
                    fig = px.violin(df, x=x_axis, y=y_axis, color=color_option,
                                    box=True, points="all",
                                    facet_row=facet_row, facet_col=facet_col)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating chart: {e}")
    else:
        st.write("Not enough numeric columns available for visualization.")
    st.subheader("Interactive Correlation Matrix")
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        corr = num_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
        col_pair = st.columns(2)
        with col_pair[0]:
            var1 = st.selectbox("Variable 1", numeric_columns, key="corr_var1")
        with col_pair[1]:
            var2 = st.selectbox("Variable 2", numeric_columns, key="corr_var2")
        if st.button("Show Scatter for Selected Variables"):
            scatter_fig = px.scatter(df, x=var1, y=var2)
            st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        st.write("No numeric data for correlation analysis.")

# -------------------------------------------------------------------
# Tab 4: SQL Query Editor
# -------------------------------------------------------------------
with tabs[3]:
    st.header("SQL Query Editor")
    st.write("Write an SQL query using `df` as the table name (e.g., SELECT * FROM df LIMIT 10):")
    query = st_ace(language="sql", theme="monokai", key="sql_editor", height=200,
                   placeholder="e.g., SELECT * FROM df LIMIT 10")
    if st.button("Run SQL Query", key="run_sql"):
        try:
            result = sqldf(query, {"df": df})
            st.session_state.sql_history.append(query)
            st.write("Query Result:")
            st.dataframe(result)
        except Exception as e:
            st.error(f"SQL Query Error: {e}")
    if st.checkbox("Show Query History"):
        st.write(st.session_state.sql_history)

# -------------------------------------------------------------------
# Tab 5: Data Lineage & Transformations
# -------------------------------------------------------------------
with tabs[4]:
    st.header("Data Lineage & Transformations")
    st.write("Log your column transformation steps and visualize the lineage.")
    with st.form("lineage_form"):
        trans_type = st.selectbox("Transformation Type", ["Rename", "Create Derived Column", "Drop Column"], key="trans_type")
        if trans_type in ["Rename", "Drop Column"]:
            source = st.selectbox("Select Column", df.columns.tolist(), key="lineage_source")
        else:
            source = st.multiselect("Select Source Columns", df.columns.tolist(), key="lineage_sources")
        target = st.text_input("New Column Name", key="lineage_target")
        submitted = st.form_submit_button("Add Transformation")
        if submitted:
            if not target:
                st.error("Please provide a target column name.")
            elif trans_type in ["Rename", "Drop Column"] and not source:
                st.error("Please select a source column.")
            else:
                step = {"type": trans_type, "source": source, "target": target}
                st.session_state.lineage_steps.append(step)
                st.success(f"Added transformation: {trans_type} {source} → {target}")
    if st.session_state.lineage_steps:
        st.write("**Transformation Log:**")
        for i, step in enumerate(st.session_state.lineage_steps, 1):
            st.write(f"{i}. {step['type']}: {step['source']} → {step['target']}")
    if st.button("Clear All Lineage Steps"):
        st.session_state.lineage_steps = []
        st.success("Lineage steps cleared.")
    dot = graphviz.Digraph()
    for col in df.columns:
        dot.node(col, col)
    for step in st.session_state.lineage_steps:
        if step["type"] in ["Rename", "Create Derived Column"]:
            if isinstance(step["source"], list):
                for src in step["source"]:
                    dot.edge(src, step["target"], label=step["type"])
            else:
                dot.edge(step["source"], step["target"], label=step["type"])
        elif step["type"] == "Drop Column":
            dot.node(step["target"], step["target"], style="filled", fillcolor="red")
            dot.edge(step["target"], "Dropped", label="Dropped")
    dot.node("Dropped", "Dropped", style="filled", fillcolor="lightgray")
    st.subheader("Lineage Graph")
    st.graphviz_chart(dot)

# -------------------------------------------------------------------
# Tab 6: Pivot Table Builder
# -------------------------------------------------------------------
with tabs[5]:
    st.header("Pivot Table Builder")
    available_cols = df.columns.tolist()
    index_col = st.selectbox("Select Index (rows)", available_cols, key="pivot_index")
    col_col = st.selectbox("Select Column (optional)", [None] + available_cols, key="pivot_columns")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.write("No numeric columns available for pivot table values.")
    else:
        value_col = st.selectbox("Select Value Column (numeric)", num_cols, key="pivot_value")
        agg_func = st.selectbox("Aggregation Function", ["sum", "mean", "count", "min", "max"], key="pivot_agg")
        if st.button("Generate Pivot Table", key="gen_pivot"):
            try:
                pivot = pd.pivot_table(df, index=index_col, columns=col_col if col_col else None,
                                       values=value_col, aggfunc=agg_func)
                st.dataframe(pivot)
                if np.issubdtype(pivot.values.dtype, np.number):
                    fig_pivot = px.imshow(pivot, text_auto=True, aspect="auto", title="Pivot Table Heatmap")
                    st.plotly_chart(fig_pivot, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating pivot table: {e}")

# -------------------------------------------------------------------
# Tab 7: Advanced Analytics
# -------------------------------------------------------------------
with tabs[6]:
    st.header("Advanced Analytics")
    st.subheader("Anomaly Detection (IQR Method)")
    anomaly_results = {}
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        anomaly_results[col] = anomalies.shape[0]
    st.dataframe(pd.DataFrame(anomaly_results, index=["Anomaly Count"]).T)
    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        corr = num_df.corr()
        fig_corr = go.Figure(data=go.Heatmap(z=corr.values,
                                              x=corr.columns,
                                              y=corr.index,
                                              colorscale="Viridis",
                                              zmin=-1, zmax=1,
                                              colorbar=dict(title="Correlation")))
        fig_corr.update_layout(title="Correlation Heatmap",
                               xaxis_title="Features",
                               yaxis_title="Features")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.write("No numeric columns available for correlation analysis.")

# -------------------------------------------------------------------
# Tab 8: Custom Dashboard
# -------------------------------------------------------------------
with tabs[7]:
    st.header("Custom Dashboard")
    st.write("Build custom metric cards for key statistics.")
    metric_options = ["Mean", "Median", "Standard Deviation", "Count", "Min", "Max"]
    selected_metric = st.selectbox("Select Metric", metric_options, key="dash_metric")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        column_to_analyze = st.selectbox("Select Column", numeric_cols, key="dash_column")
        if st.button("Show Metric Card", key="show_metric"):
            if column_to_analyze:
                if selected_metric == "Mean":
                    value = df[column_to_analyze].mean()
                elif selected_metric == "Median":
                    value = df[column_to_analyze].median()
                elif selected_metric == "Standard Deviation":
                    value = df[column_to_analyze].std()
                elif selected_metric == "Count":
                    value = df[column_to_analyze].count()
                elif selected_metric == "Min":
                    value = df[column_to_analyze].min()
                elif selected_metric == "Max":
                    value = df[column_to_analyze].max()
                st.markdown(f"""
                <div class="card">
                    <h3>{selected_metric} of {column_to_analyze}</h3>
                    <p style="font-size: 28px; font-weight: bold;">{value:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Please select a numeric column.")
    else:
        st.write("No numeric columns available for dashboard metrics.")

# -------------------------------------------------------------------
# Tab 9: Data Merge
# -------------------------------------------------------------------
with tabs[8]:
    st.header("Data Merge")
    if df2 is None:
        st.write("No secondary data uploaded. Use the sidebar to upload a secondary data file.")
    else:
        st.write("Merge the primary and secondary datasets.")
        merge_type = st.selectbox("Merge Type", ["inner", "left", "right", "outer"], key="merge_type")
        primary_key = st.selectbox("Select Primary Key", df.columns.tolist(), key="merge_key1")
        secondary_key = st.selectbox("Select Secondary Key", df2.columns.tolist(), key="merge_key2")
        if st.button("Merge Datasets"):
            try:
                merged_df = pd.merge(df, df2, left_on=primary_key, right_on=secondary_key, how=merge_type)
                st.success(f"Merged dataset shape: {merged_df.shape}")
                if use_aggrid:
                    gb = GridOptionsBuilder.from_dataframe(merged_df)
                    gb.configure_pagination(enabled=True, paginationAutoPageSize=True)
                    gb.configure_side_bar()
                    grid_options = gb.build()
                    AgGrid(merged_df, gridOptions=grid_options, height=500, theme="streamlit")
                else:
                    st.dataframe(merged_df)
                csv_data = merged_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Merged Data as CSV", data=csv_data, file_name="merged_data.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error merging datasets: {e}")

# -------------------------------------------------------------------
# Tab 10: Google Drive Integration
# -------------------------------------------------------------------
with tabs[9]:
    st.header("Google Drive Integration")
    st.write("Authenticate with your Google Drive account and manage files.")
    if not use_gdrive:
        st.error("PyDrive2 is not installed. Google Drive integration is not available.")
    else:
        if st.button("Authenticate with Google Drive"):
            try:
                drive = authenticate_drive()
                st.session_state.drive = drive
                st.success("Authenticated with Google Drive!")
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        if "drive" in st.session_state:
            drive = st.session_state.drive
            if st.button("List Files in Root Folder"):
                try:
                    files = list_drive_files(drive, "root")
                    for f in files:
                        st.write(f"Title: {f['title']}  |  ID: {f['id']}")
                except Exception as e:
                    st.error(f"Error listing files: {e}")
            uploaded_to_drive = st.file_uploader("Select file to upload to Google Drive", key="gdrive_upload")
            if uploaded_to_drive:
                try:
                    with open(uploaded_to_drive.name, "wb") as f:
                        f.write(uploaded_to_drive.getbuffer())
                    file_id = upload_file_to_drive(drive, uploaded_to_drive.name, uploaded_to_drive.name)
                    st.success(f"File uploaded to Google Drive with ID: {file_id}")
                except Exception as e:
                    st.error(f"Error uploading file: {e}")
        else:
            st.info("Please authenticate with Google Drive first.")

