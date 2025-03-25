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
import graphviz

# Page configuration and custom CSS styling for an advanced, modern look.
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

#############################################
# Helper Functions
#############################################

def process_file(uploaded_file):
    """Read CSV, Excel, or Parquet file into a dict of DataFrames."""
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

#############################################
# Sidebar: File Upload & Sheet Selection
#############################################

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

#############################################
# Initialize Session State for Data Lineage
#############################################

if "lineage_steps" not in st.session_state:
    st.session_state.lineage_steps = []

#############################################
# Create Tabs for Advanced Features
#############################################

tabs = st.tabs([
    "Data Preview", "Summary & Cleaning", "Visualizations", "SQL Query",
    "Data Lineage", "Pivot Table Builder", "Advanced Analytics", "Custom Dashboard"
])

#############################################
# Tab 1: Data Preview & Filtering
#############################################

with tabs[0]:
    st.header("Data Preview & Filtering")
    st.write("Interactively select columns and search within the data.")
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

#############################################
# Tab 2: Data Summary & Cleaning
#############################################

with tabs[1]:
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

#############################################
# Tab 3: Interactive Visualizations
#############################################

with tabs[2]:
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
        facet_row = st.selectbox("Facet Row (optional)", [None] + categorical_columns, key="facet_row")
        facet_col = st.selectbox("Facet Column (optional)", [None] + categorical_columns, key="facet_col")
        if st.button("Generate Chart"):
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

#############################################
# Tab 4: SQL Query Editor
#############################################

with tabs[3]:
    st.header("SQL Query Editor")
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

#############################################
# Tab 5: Data Lineage
#############################################

with tabs[4]:
    st.header("Data Lineage")
    st.write("Log and visualize your column transformation steps.")
    # Transformation form
    with st.form("lineage_form"):
        trans_type = st.selectbox("Transformation Type", ["Rename", "Create Derived Column", "Drop Column"])
        if trans_type in ["Rename", "Drop Column"]:
            source = [st.selectbox("Select Column", df.columns.tolist(), key="lineage_source")]
        else:  # Create Derived Column
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
    # Display current lineage steps
    if st.session_state.lineage_steps:
        st.write("**Transformation Log:**")
        for i, step in enumerate(st.session_state.lineage_steps, 1):
            st.write(f"{i}. {step['type']}: {step['source']} → {step['target']}")
    if st.button("Clear All Lineage Steps"):
        st.session_state.lineage_steps = []
        st.success("Lineage steps cleared.")
    # Build lineage graph using graphviz
    dot = graphviz.Digraph()
    # Add nodes for original columns
    for col in df.columns:
        dot.node(col, col)
    # Add nodes/edges from transformations
    for step in st.session_state.lineage_steps:
        if step["type"] in ["Rename", "Create Derived Column"]:
            for src in step["source"]:
                dot.edge(src, step["target"], label=step["type"])
        elif step["type"] == "Drop Column":
            dot.node(step["target"], step["target"], style="filled", fillcolor="red")
            dot.edge(step["target"], "Dropped", label="Dropped")
    dot.node("Dropped", "Dropped", style="filled", fillcolor="lightgray")
    st.subheader("Lineage Graph")
    st.graphviz_chart(dot)

#############################################
# Tab 6: Pivot Table Builder
#############################################

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
        if st.button("Generate Pivot Table"):
            try:
                pivot = pd.pivot_table(df, index=index_col, columns=col_col if col_col else None,
                                       values=value_col, aggfunc=agg_func)
                st.dataframe(pivot)
                if np.issubdtype(pivot.values.dtype, np.number):
                    fig_pivot = px.imshow(pivot, text_auto=True, aspect="auto", title="Pivot Table Heatmap")
                    st.plotly_chart(fig_pivot, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating pivot table: {e}")

#############################################
# Tab 7: Advanced Analytics
#############################################

with tabs[6]:
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

#############################################
# Tab 8: Custom Dashboard
#############################################

with tabs[7]:
    st.header("Custom Dashboard")
    st.write("Create custom metric cards for key statistics.")
    metric_options = ["Mean", "Median", "Standard Deviation", "Count", "Min", "Max"]
    selected_metric = st.selectbox("Select Metric", metric_options, key="dash_metric")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        column_to_analyze = st.selectbox("Select Column", numeric_cols, key="dash_column")
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
