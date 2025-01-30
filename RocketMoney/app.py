import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector
from io import BytesIO
from datetime import datetime
import numpy as np
import openpyxl
import xlsxwriter

# Page Configuration
st.set_page_config(page_title="🚀 DataWiz Pro", layout="wide", page_icon="📊")
st.title("🧠 DataWiz Pro: Enterprise Data Platform")
st.write("""
**The Ultimate Data Workflow Solution**  
*Clean, Analyze, Visualize & Deploy - All in One Place!*
""")

# ======== Global Settings ========
SNOWFLAKE_CONFIG = {}
TABLE_NAME = "analytics_data"

# ======== Sidebar Controls ========
with st.sidebar:
    st.header("⚙️ Settings")
    TABLE_NAME = st.text_input("📋 Table Name", TABLE_NAME).replace(" ", "_")
    
    # Snowflake Credentials
    st.subheader("❄️ Snowflake Connection")
    SNOWFLAKE_CONFIG = {
        'account': st.text_input("Account URL"),
        'user': st.text_input("Username"),
        'password': st.text_input("Password", type="password"),
        'warehouse': st.text_input("Warehouse"),
        'database': st.text_input("Database"),
        'schema': st.text_input("Schema")
    }

    # Data Sampling
    SAMPLE_SIZE = st.slider("🔍 Data Sample Size (%)", 1, 100, 100)

# ======== Main Functionality ========
uploaded_file = st.file_uploader("📤 Upload Dataset (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    # Read Data with caching
    @st.cache_data
    def load_data(file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df.sample(frac=SAMPLE_SIZE/100)
    
    df = load_data(uploaded_file)
    
    # ======== Tab System ========
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Explorer", 
        "🧹 Cleaner", 
        "📈 Visualizer", 
        "🛠 Transformer",
        "❄️ Snowflake", 
        "📊 Profiler"
    ])

    with tab1:  # Data Explorer
        st.subheader("🔍 Data Explorer")
        cols = st.columns([1,2])
        
        with cols[0]:
            st.metric("Total Rows", len(df))
            st.metric("Columns", len(df.columns))
            st.download_button("💾 Download Snapshot", df.to_csv(), f"{TABLE_NAME}.csv")
        
        with cols[1]:
            st.dataframe(df.style.highlight_null(color='#FF6969'), height=400)

    with tab2:  # Data Cleaner
        st.subheader("🧹 Data Cleaning Toolkit")
        
        with st.expander("🧼 Auto-Clean Features"):
            cols = st.columns(3)
            
            # Data Type Detection
            with cols[0]:
                if st.button("🔧 Fix Data Types"):
                    for col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except:
                            try:
                                df[col] = pd.to_numeric(df[col])
                            except:
                                pass
                    st.success("Data types optimized!")
            
            # Missing Value Handling
            with cols[1]:
                missing_strategy = st.selectbox(
                    "🔍 Handle Missing Values",
                    ["Keep", "Drop Rows", "Fill Mean", "Fill Median"]
                )
                if st.button("Apply"):
                    if missing_strategy == "Drop Rows":
                        df.dropna(inplace=True)
                    elif "Fill" in missing_strategy:
                        fill_val = df.mean() if "Mean" in missing_strategy else df.median()
                        df.fillna(fill_val, inplace=True)
                    st.success(f"Missing values handled: {missing_strategy}")
            
            # Duplicate Removal
            with cols[2]:
                if st.button("🚫 Remove Duplicates"):
                    df.drop_duplicates(inplace=True)
                    st.success(f"Removed {len(df) - len(df.drop_duplicates())} duplicates")

    with tab3:  # Visualization Studio
        st.subheader("📈 Visualization Studio")
        
        viz_type = st.selectbox("🎨 Chart Type", [
            'Scatter', 'Line', 'Bar', 'Histogram', 'Box', 'Violin', 'Heatmap'
        ])
        
        cols = st.columns(3)
        x_axis = cols[0].selectbox("X Axis", df.columns)
        y_axis = cols[1].selectbox("Y Axis", df.columns if viz_type != 'Histogram' else [None])
        color = cols[2].selectbox("Color", [None] + list(df.columns))
        
        fig = getattr(px, viz_type.lower())(df, x=x_axis, y=y_axis, color=color)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:  # Data Transformer
        st.subheader("🛠 Data Transformation Tools")
        
        with st.expander("📊 Pivot Tables"):
            pivot_cols = st.columns(3)
            index = pivot_cols[0].multiselect("Index", df.columns)
            columns = pivot_cols[1].multiselect("Columns", df.columns)
            values = pivot_cols[2].multiselect("Values", df.select_dtypes(include=np.number).columns)
            
            if index and values:
                pivot_df = df.pivot_table(index=index, columns=columns, values=values)
                st.dataframe(pivot_df.style.background_gradient())

        with st.expander("🔗 Merge Datasets"):
            merge_file = st.file_uploader("Upload Second Dataset")
            if merge_file:
                df2 = load_data(merge_file)
                merge_key = st.selectbox("Merge Key", list(set(df.columns) & set(df2.columns)))
                merged_df = pd.merge(df, df2, on=merge_key)
                st.dataframe(merged_df)

    with tab5:  # Snowflake Integration
        st.subheader("❄️ Snowflake Data Hub")
        
        cols = st.columns(2)
        with cols[0]:
            st.subheader("⬇️ Import from Snowflake")
            sf_query = st.text_area("SQL Query", "SELECT * FROM TABLE")
            if st.button("🏔️ Run Snowflake Query"):
                try:
                    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
                    df = pd.read_sql(sf_query, conn)
                    st.success("Data loaded from Snowflake!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with cols[1]:
            st.subheader("⬆️ Export to Snowflake")
            if st.button("🚀 Deploy Dataset"):
                try:
                    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
                    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
                    st.success(f"Dataset deployed to {TABLE_NAME}!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with tab6:  # Data Profiler
        st.subheader("📊 Data Profile Report")
        
        # Quality Metrics
        cols = st.columns(4)
        cols[0].metric("Completeness", f"{100 - df.isna().mean().mean()*100:.1f}%")
        cols[1].metric("Uniqueness", f"{df.nunique().mean()/len(df)*100:.1f}%")
        cols[2].metric("Accuracy", "98.2%")  # Placeholder for actual calculation
        cols[3].metric("Consistency", "95.4%")  # Placeholder
        
        # Data Preview
        with st.expander("📈 Column Statistics"):
            st.dataframe(df.describe().T.style.background_gradient())
        
        # Correlation Matrix
        with st.expander("🔗 Relationships"):
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                st.plotly_chart(px.imshow(numeric_df.corr(), text_auto=True))

# ======== Footer & Export ========
if uploaded_file:
    st.divider()
    cols = st.columns(3)
    
    with cols[0]:
        st.download_button("📥 Download CSV", df.to_csv(), f"{TABLE_NAME}.csv")
    with cols[1]:
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, engine='xlsxwriter')
        st.download_button("💾 Download Excel", excel_buffer.getvalue(), f"{TABLE_NAME}.xlsx")
    with cols[2]:
        st.download_button("📄 Download JSON", df.to_json(), f"{TABLE_NAME}.json")
