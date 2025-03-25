import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Try to import st_aggrid; if unavailable, fall back to st.dataframe.
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    use_aggrid = True
except ImportError:
    use_aggrid = False

from streamlit_ace import st_ace
from pandasql import sqldf

# Page configuration and custom CSS for an advanced, modern look.
st.set_page_config(page_title="Ultimate Data Analyzer", layout="wide")
st.markdown("""
<style>
/* Backgrounds */
.reportview-container { background: #f7f9fc; }
.sidebar .sidebar-content { background: #ffffff; }
/* Card styling for dashboard metrics */
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    margin: 10px;
}
.card h3 { margin: 0 0 10px 0; }
</style>
""", unsafe_allow_html=True)

# Function to process uploaded files.
def process_file(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    try:
        if ext == ".csv":
            return {"Sheet1": pd.read_csv(uploaded_file)}
        elif ext == ".xlsx":
            # Read all sheets from an Excel file.
            return pd.read_excel(uploaded_file, sheet_name=None)
        elif ext == ".parquet":
            return {"Sheet1": pd.read_parquet(uploaded_file)}
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Sidebar: Data upload and sheet selection.
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV, Excel, or Parquet", type=["csv", "xlsx", "parquet"])
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

# Create tabs for advanced features.
(tab_preview, tab_summary, tab_viz, tab_sql, tab_analytics, tab_dashboard) = st.tabs([
    "Data Preview", "Summary & Cleaning", "Visualizations", "SQL Query", "Advanced Analytics", "Custom Dashboard"
])

# --- Tab 1: Data Preview & Filtering ---
with tab_preview:
    st.header("Data Preview & Filtering")
    st.write("Preview your data with interactive column selection and row filtering.")
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to display", all_columns, default=all_columns)
    search_query = st.text_input("Search (case-insensitive on selected columns)", "")
    df_display = df[selected_columns] if selected_columns else df
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

# --- Tab 2: Data Summary & Cleaning ---
with tab_summary:
    st.header("Data Summary & Cleaning")
    st.write("**Shape:**", df.shape)
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
    
    if st.button("Drop Duplicates"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicates dropped.")

# --- Tab 3: Interactive Visualizations ---
with tab_viz:
    st.header("Interactive Visualizations")
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
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin"], key="chart_type")
        # Advanced: Option for facetting
        facet_row = st.selectbox("Facet Row (optional)", [None] + categorical_columns, key="facet_row")
        facet_col = st.selectbox("Facet Column (optional)", [None] + categorical_columns, key="facet_col")
        if st.button("Generate Chart"):
            try:
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option, facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_option, facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_option, facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, color=color_option, facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Box":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_option, facet_row=facet_row, facet_col=facet_col)
                elif chart_type == "Violin":
                    fig = px.violin(df, x=x_axis, y=y_axis, color=color_option, box=True, points="all",
                                    facet_row=facet_row, facet_col=facet_col)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating chart: {e}")
    else:
        st.write("Not enough numeric columns available for visualization.")

# --- Tab 4: SQL Query ---
with tab_sql:
    st.header("SQL Query")
    st.write("Write an SQL query using `df` as the table name (e.g., SELECT * FROM df LIMIT 10):")
    query = st_ace(language="sql", theme="monokai", key="sql_editor", height=200,
                   placeholder="e.g., SELECT * FROM df LIMIT 10")
    if st.button("Run SQL Query"):
        try:
            result = sqldf(query, {"df": df})
            st.write("Query Result:")
            st.dataframe(result)
        except Exception as e:
            st.error(f"SQL Query Error: {e}")

# --- Tab 5: Advanced Analytics ---
with tab_analytics:
    st.header("Advanced Analytics")
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

# --- Tab 6: Custom Dashboard ---
with tab_dashboard:
    st.header("Custom Dashboard")
    st.write("Create custom metric cards for key statistics.")
    metric_options = ["Mean", "Median", "Standard Deviation", "Count", "Min", "Max"]
    selected_metric = st.selectbox("Select Metric", metric_options)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        column_to_analyze = st.selectbox("Select Column", numeric_cols)
        if st.button("Show Metric Card"):
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
                    <p style="font-size: 24px; font-weight: bold;">{value:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Please select a numeric column.")
    else:
        st.write("No numeric columns available for dashboard metrics.")
