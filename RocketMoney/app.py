import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import graphviz
from streamlit_ace import st_ace
from pandasql import sqldf
from io import BytesIO
import zipfile

# Additional UI libraries
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

# Optional interactive grid: if not available, fall back to st.dataframe.
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    use_aggrid = True
except ImportError:
    use_aggrid = False

# -------------------------------------------------------------------
# Page Configuration & Custom Styling
# -------------------------------------------------------------------
st.set_page_config(page_title="Ultimate Data Analyzer Pro", layout="wide")
st.markdown("""
<style>
  body {
      background: linear-gradient(to right, #ece9e6, #ffffff);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  .sidebar .sidebar-content { background: #ffffff; }
  /* Card styling for dashboard metrics */
  .card {
      background-color: #ffffff;
      padding: 20px;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      margin: 10px;
  }
  .card h3 { margin: 0 0 10px 0; color: #333; }
  .field-list { background: #eef; padding: 8px; border-radius: 8px; margin-top: 5px; }
  .header-title { font-size: 2rem; font-weight: 600; text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar: Option Menu for Navigation & Data Upload
# -------------------------------------------------------------------
with st.sidebar:
    choice = option_menu("Main Menu", ["Upload Data", "Settings"], 
                         icons=["upload", "gear-fill"], menu_icon="app-indicator", default_index=0)
    
    if choice == "Upload Data":
        st.header("Upload Primary Data")
        uploaded_file = st.file_uploader("Choose CSV, Excel, or Parquet", type=["csv", "xlsx", "parquet"])
        if uploaded_file:
            def process_file(uploaded_file):
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
            data = process_file(uploaded_file)
            if data is not None:
                sheet_names = list(data.keys())
                sheet_selected = st.selectbox("Select Sheet", sheet_names)
                df = data[sheet_selected]
                st.session_state.df = df.copy()
                st.success("Data uploaded successfully!")
        else:
            st.info("Please upload a file.")
    elif choice == "Settings":
        st.header("Settings")
        dark_mode = st.checkbox("Enable Dark Mode", value=False)
        if dark_mode:
            st.markdown("""
            <style>
              body { background: #1e1e1e; color: #ccc; }
              .card { background-color: #333; color: #eee; }
            </style>
            """, unsafe_allow_html=True)

# If data is not yet uploaded, stop.
if "df" not in st.session_state:
    st.stop()

# Use the uploaded data.
df = st.session_state.df

# -------------------------------------------------------------------
# Initialize Session State for Data Lineage & SQL History
# -------------------------------------------------------------------
if "lineage_steps" not in st.session_state:
    st.session_state.lineage_steps = []
if "sql_history" not in st.session_state:
    st.session_state.sql_history = []

# -------------------------------------------------------------------
# Main Tabs for Features
# -------------------------------------------------------------------
tabs = st.tabs([
    "Data Preview & Filtering",
    "Summary & Profiling",
    "Data Profile",
    "Advanced Visualizations",
    "SQL Query Editor",
    "Data Lineage & Transformations",
    "Pivot Table Builder",
    "Advanced Analytics",
    "Custom Dashboard"
])

# Always update df from session state.
df = st.session_state.df

# -------------------------------------------------------------------
# Tab 1: Data Preview & Filtering
# -------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='header-title'>Data Preview & Filtering</div>", unsafe_allow_html=True)
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to display", all_columns, default=all_columns)
    df_display = df[selected_columns] if selected_columns else df
    with st.expander("Add Filters"):
        for col in df_display.columns:
            if pd.api.types.is_numeric_dtype(df_display[col]):
                min_val, max_val = float(df_display[col].min()), float(df_display[col].max())
                condition = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
                df_display = df_display[(df_display[col] >= condition[0]) & (df_display[col] <= condition[1])]
            elif pd.api.types.is_string_dtype(df_display[col]):
                unique_vals = df_display[col].dropna().unique().tolist()
                if unique_vals:
                    condition = st.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
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
    st.markdown("<div class='header-title'>Summary & Profiling</div>", unsafe_allow_html=True)
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
# Tab 3: Data Profile (YData Profiling)
# -------------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='header-title'>Data Profile Report</div>", unsafe_allow_html=True)
    if st.button("Generate Profile Report"):
        profile = ProfileReport(df, explorative=True)
        profile_html = profile.to_html()
        components.html(profile_html, height=800, scrolling=True)

# -------------------------------------------------------------------
# Tab 4: Advanced Visualizations
# -------------------------------------------------------------------
with tabs[3]:
    st.markdown("<div class='header-title'>Advanced Visualizations</div>", unsafe_allow_html=True)
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
# Tab 5: Data Lineage & Transformations
# -------------------------------------------------------------------
with tabs[4]:
    st.markdown("<div class='header-title'>Data Lineage & Transformations</div>", unsafe_allow_html=True)
    st.write("Log your transformation steps and preview the updated data.")
    with st.form("lineage_form"):
        trans_type = st.selectbox("Transformation Type", ["Rename", "Create Derived Column", "Drop Column"], key="trans_type")
        if trans_type in ["Rename", "Drop Column"]:
            source = st.selectbox("Select Column", df.columns.tolist(), key="lineage_source")
            formula = None
        else:
            st.write("Enter a formula using existing column names (e.g., col1 + col2 * 2).")
            formula = st_ace(
                language="python",
                theme="monokai",
                key="calc_formula",
                height=100,
                placeholder="e.g., col1 + col2 * 2",
                options={"enableBasicAutocompletion": True, "enableLiveAutocompletion": True}
            )
            source = None
            st.markdown("<div class='field-list'><strong>Available fields:</strong> " + ", ".join(df.columns.tolist()) + "</div>", unsafe_allow_html=True)
        target = st.text_input("New Column Name", key="lineage_target")
        submitted = st.form_submit_button("Add Transformation")
        if submitted:
            if not target:
                st.error("Please provide a target column name.")
            elif trans_type == "Create Derived Column" and not formula.strip():
                st.error("Please provide a formula for the derived column.")
            else:
                if trans_type == "Rename":
                    st.session_state.df = st.session_state.df.rename(columns={source: target})
                elif trans_type == "Drop Column":
                    st.session_state.df = st.session_state.df.drop(columns=[source])
                elif trans_type == "Create Derived Column":
                    try:
                        st.session_state.df[target] = st.session_state.df.eval(formula)
                    except Exception as e:
                        st.error(f"Error computing derived column: {e}")
                        st.stop()
                step = {"type": trans_type, "source": source if source else formula, "target": target}
                st.session_state.lineage_steps.append(step)
                st.success(f"Transformation applied: {trans_type} → {target}")
    df = st.session_state.df
    if st.session_state.lineage_steps:
        st.write("**Transformation Log:**")
        st.dataframe(pd.DataFrame(st.session_state.lineage_steps))
    if st.button("Clear All Lineage Steps"):
        st.session_state.lineage_steps = []
        st.success("Lineage steps cleared.")
    dot = graphviz.Digraph()
    for col in df.columns:
        dot.node(col, col)
    for step in st.session_state.lineage_steps:
        if step["type"] in ["Rename", "Create Derived Column"]:
            src_label = step["source"] if isinstance(step["source"], str) else ", ".join(step["source"])
            dot.edge(src_label, step["target"], label=step["type"])
        elif step["type"] == "Drop Column":
            dot.node(step["target"], step["target"], style="filled", fillcolor="red")
            dot.edge(step["target"], "Dropped", label="Dropped")
    dot.node("Dropped", "Dropped", style="filled", fillcolor="lightgray")
    st.subheader("Lineage Graph")
    st.graphviz_chart(dot)
    st.subheader("Preview & Download Updated Data")
    st.dataframe(df)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_data, file_name="updated_data.csv", mime="text/csv")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("updated_data.csv", csv_data)
    zip_buffer.seek(0)
    st.download_button("Download Compressed ZIP", data=zip_buffer, file_name="updated_data.zip", mime="application/zip")

# -------------------------------------------------------------------
# Tab 6: Pivot Table Builder
# -------------------------------------------------------------------
with tabs[5]:
    st.markdown("<div class='header-title'>Pivot Table Builder</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='header-title'>Advanced Analytics</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='header-title'>Custom Dashboard</div>", unsafe_allow_html=True)
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
