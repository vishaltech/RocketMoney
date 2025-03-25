import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import graphviz
from io import BytesIO
import zipfile
from pathlib import Path
from streamlit_ace import st_ace

# -----------------------------
# Minimal Requirements
# -----------------------------
# - streamlit
# - pandas
# - numpy
# - plotly
# - streamlit-ace
# - openpyxl
# - graphviz
# - pyarrow

st.set_page_config(page_title="Minimal Data Analyzer", layout="wide")

# ----------------------------------------
# Session State Initialization
# ----------------------------------------
if "df_original" not in st.session_state:
    st.session_state.df_original = None  # The unmodified data
if "df" not in st.session_state:
    st.session_state.df = None  # The currently filtered/transformed data
if "lineage" not in st.session_state:
    st.session_state.lineage = []  # List of transformation steps

# ----------------------------------------
# Upload Section
# ----------------------------------------
st.title("Minimal Data Analyzer")

uploaded_file = st.file_uploader("Upload a CSV, Excel, or Parquet file", type=["csv", "xlsx", "parquet"])
if uploaded_file:
    ext = Path(uploaded_file.name).suffix.lower()
    try:
        if ext == ".csv":
            df_tmp = pd.read_csv(uploaded_file)
        elif ext == ".xlsx":
            # If multiple sheets, just take the first
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            first_sheet = list(all_sheets.keys())[0]
            df_tmp = all_sheets[first_sheet]
        elif ext == ".parquet":
            df_tmp = pd.read_parquet(uploaded_file)
        st.session_state.df_original = df_tmp.copy()
        st.session_state.df = df_tmp.copy()
        st.session_state.lineage = []
        st.success("File uploaded and data loaded!")
    except Exception as e:
        st.error(f"Error loading file: {e}")

if st.session_state.df is None:
    st.stop()

df = st.session_state.df

# ----------------------------------------
# Tabs for Features
# ----------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data", "Filter", "Transform", "Visualize", "Download"
])

# ----------------------------------------
# Tab 1: Data Preview
# ----------------------------------------
with tab1:
    st.subheader("Data Preview")
    st.dataframe(df.reset_index(drop=True))  # Show data with index reset

# ----------------------------------------
# Tab 2: Filtering
# ----------------------------------------
with tab2:
    st.subheader("Filtering")

    # Provide a "Reset Filter" button
    if st.button("Reset to Original Data"):
        st.session_state.df = st.session_state.df_original.copy()
        st.session_state.lineage = []
        st.success("Data reset to original file contents!")
        st.experimental_rerun()

    st.write("Select a column to filter on:")
    column_to_filter = st.selectbox("Column", df.columns.tolist())

    if column_to_filter:
        col_data = df[column_to_filter]
        # If numeric, show a slider
        if pd.api.types.is_numeric_dtype(col_data):
            min_val, max_val = float(col_data.min()), float(col_data.max())
            chosen_range = st.slider(
                "Filter range",
                min_val,
                max_val,
                (min_val, max_val)
            )
            if st.button("Apply Numeric Filter"):
                st.session_state.df = st.session_state.df[
                    (st.session_state.df[column_to_filter] >= chosen_range[0]) &
                    (st.session_state.df[column_to_filter] <= chosen_range[1])
                ]
                # Log the step
                st.session_state.lineage.append({
                    "type": "Filter (Numeric)",
                    "column": column_to_filter,
                    "range": chosen_range
                })
                st.success(f"Filtered {column_to_filter} to between {chosen_range[0]} and {chosen_range[1]}")
                st.experimental_rerun()
        else:
            # Categorical or string data
            unique_vals = col_data.dropna().unique().tolist()
            chosen_vals = st.multiselect("Values to keep", unique_vals, default=unique_vals)
            if st.button("Apply Categorical Filter"):
                st.session_state.df = st.session_state.df[st.session_state.df[column_to_filter].isin(chosen_vals)]
                st.session_state.lineage.append({
                    "type": "Filter (Categorical)",
                    "column": column_to_filter,
                    "values": chosen_vals
                })
                st.success(f"Filtered {column_to_filter} to {chosen_vals}")
                st.experimental_rerun()

    st.write("Filtered Data Preview:")
    st.dataframe(st.session_state.df.reset_index(drop=True))

# ----------------------------------------
# Tab 3: Transform
# ----------------------------------------
with tab3:
    st.subheader("Transform Data (Rename, Drop, or Derived Column)")

    # Show the lineage steps so far
    if st.session_state.lineage:
        st.write("**Transformation Log:**")
        st.dataframe(pd.DataFrame(st.session_state.lineage))
    else:
        st.write("No transformations applied yet.")

    # Minimal lineage graph
    dot = graphviz.Digraph()
    # Make nodes for each column
    for c in df.columns:
        dot.node(c, c)

    for step in st.session_state.lineage:
        if step["type"] in ["Rename", "Create Derived Column"]:
            src = step.get("source") or step.get("formula", "unknown")
            tgt = step["target"]
            dot.edge(str(src), str(tgt), label=step["type"])
        elif step["type"] == "Drop":
            col = step["column"]
            dot.node(col, col, style="filled", fillcolor="red")
            dot.edge(col, "Dropped", label="Drop")
    dot.node("Dropped", "Dropped", style="filled", fillcolor="lightgray")

    st.graphviz_chart(dot)

    # A form for transformations
    with st.form("transform_form"):
        transform_type = st.selectbox("Select a Transformation", ["Rename", "Drop", "Create Derived Column"])

        if transform_type == "Rename":
            col_to_rename = st.selectbox("Column to rename", df.columns.tolist())
            new_name = st.text_input("New column name")
        elif transform_type == "Drop":
            col_to_drop = st.selectbox("Column to drop", df.columns.tolist())
        else:  # Create Derived Column
            st.write("Use existing columns in an expression, e.g. `col1 + col2 * 2`")
            formula = st_ace(
                language="python",
                theme="monokai",
                placeholder="e.g. col1 + col2 * 2",
                key="derived_formula",
                height=100,
                options={"enableBasicAutocompletion": True, "enableLiveAutocompletion": True}
            )
            new_col_name = st.text_input("New column name for derived expression")

        submit_transform = st.form_submit_button("Apply Transformation")

    if submit_transform:
        try:
            if transform_type == "Rename":
                if not new_name.strip():
                    st.error("Please provide a valid new name.")
                else:
                    st.session_state.df = st.session_state.df.rename(columns={col_to_rename: new_name})
                    st.session_state.lineage.append({
                        "type": "Rename",
                        "source": col_to_rename,
                        "target": new_name
                    })
                    st.success(f"Renamed {col_to_rename} to {new_name}")
                    st.experimental_rerun()
            elif transform_type == "Drop":
                st.session_state.df = st.session_state.df.drop(columns=[col_to_drop])
                st.session_state.lineage.append({
                    "type": "Drop",
                    "column": col_to_drop
                })
                st.success(f"Dropped {col_to_drop}")
                st.experimental_rerun()
            else:
                # Create Derived Column
                if not formula or not formula.strip():
                    st.error("Please provide a formula.")
                elif not new_col_name.strip():
                    st.error("Please provide a new column name.")
                else:
                    st.session_state.df[new_col_name] = st.session_state.df.eval(formula)
                    st.session_state.lineage.append({
                        "type": "Create Derived Column",
                        "source": formula,
                        "target": new_col_name
                    })
                    st.success(f"Created derived column {new_col_name} using formula: {formula}")
                    st.experimental_rerun()
        except Exception as e:
            st.error(f"Transformation error: {e}")

# ----------------------------------------
# Tab 4: Visualize
# ----------------------------------------
with tab4:
    st.subheader("Visualize Data with Plotly")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    x_choice = st.selectbox("X Axis", df.columns.tolist())
    y_choice = st.selectbox("Y Axis", df.columns.tolist())
    color_choice = st.selectbox("Color By (optional)", [None] + df.columns.tolist())
    chart_type = st.selectbox("Chart Type", ["Scatter", "Bar", "Line", "Histogram", "Box"])

    if st.button("Generate Chart"):
        try:
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_choice, y=y_choice, color=color_choice)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_choice, y=y_choice, color=color_choice)
            elif chart_type == "Line":
                fig = px.line(df, x=x_choice, y=y_choice, color=color_choice)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_choice, color=color_choice)
            else:
                fig = px.box(df, x=x_choice, y=y_choice, color=color_choice)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization error: {e}")

# ----------------------------------------
# Tab 5: Download
# ----------------------------------------
with tab5:
    st.subheader("Download Final Data")
    st.dataframe(df.reset_index(drop=True))

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name="final_data.csv",
        mime="text/csv"
    )

    # Compressed ZIP
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("final_data.csv", csv_data)
    zip_buffer.seek(0)

    st.download_button(
        label="Download as ZIP",
        data=zip_buffer,
        file_name="final_data.zip",
        mime="application/zip"
    )
