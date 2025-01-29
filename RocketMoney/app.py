import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import altair as alt
from fpdf import FPDF

# Set up the Streamlit page
st.set_page_config(page_title="🚀 Advanced Data Analyzer", layout="wide")
st.title("📊 Advanced Data Analyzer")
st.write("Upload an Excel or CSV file to explore, clean, visualize, and analyze your data!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Excel or CSV file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Display Data Preview
        st.write("### 📝 Data Preview")
        st.dataframe(df.head())

        # Column Selection
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        # 🔥 Data Cleaning
        st.write("### 🛠 Data Cleaning & Processing")
        
        if st.button("Drop Duplicates"):
            df.drop_duplicates(inplace=True)
            st.success("✅ Duplicates removed successfully!")

        if df.isnull().sum().sum() > 0:
            if st.button("Fill Missing Values with Mean"):
                df.fillna(df.select_dtypes(include=[np.number]).mean(numeric_only=True), inplace=True)
                st.success("✅ Missing values filled with column mean!")

        # 🚨 Outlier Detection
        st.write("### 🚨 Outlier Detection")
        if numeric_columns:
            outlier_column = st.selectbox("Select a column to detect outliers", numeric_columns)
            if st.button("Detect Outliers"):
                Q1 = df[outlier_column].quantile(0.25)
                Q3 = df[outlier_column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[outlier_column] < (Q1 - 1.5 * IQR)) | (df[outlier_column] > (Q3 + 1.5 * IQR))]
                st.write(f"🚨 Found {len(outliers)} outliers in `{outlier_column}`")
                st.dataframe(outliers)

        # 📊 Advanced Visualization
        st.write("### 📊 Interactive Data Visualization")
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Pie Chart"])
        x_axis = st.selectbox("Select X-axis", numeric_columns + categorical_columns)
        
        y_axis = None
        if chart_type != "Pie Chart":
            y_axis = st.selectbox("Select Y-axis", numeric_columns)

        if st.button("Generate Chart"):
            if x_axis:
                if chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"📊 Bar Chart: {y_axis} vs {x_axis}")
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"🔬 Scatter Plot: {y_axis} vs {x_axis}")
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis, title=f"📈 Line Chart: {y_axis} vs {x_axis}")
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=x_axis, title=f"🍰 Pie Chart: {x_axis}")
                else:
                    fig = px.histogram(df, x=x_axis, title=f"📊 Histogram: {x_axis}")
                
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ Please select valid columns for visualization.")

        # 🌍 Geospatial Mapping
        if "latitude" in df.columns and "longitude" in df.columns:
            st.write("### 🌍 Geospatial Data Visualization")
            fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, height=500)
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig)

        # 🔍 Dynamic Data Filtering
        st.write("### 🔍 Dynamic Data Filtering")
        selected_column = st.selectbox("Select column to filter", numeric_columns + categorical_columns)
        filter_value = st.text_input(f"Enter value to filter `{selected_column}`")

        if st.button("Apply Filter"):
            filtered_df = df[df[selected_column].astype(str).str.contains(filter_value, case=False, na=False)]
            st.write(f"Showing results for `{selected_column}` containing '{filter_value}'")
            st.dataframe(filtered_df)

        # 📑 Generate PDF Report
        st.write("### 📑 Generate Report (PDF)")
        if st.button("Download PDF Report"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, "📊 Data Report", ln=True, align="C")
            pdf.ln(10)

            # Summary Statistics
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"🔍 **Dataset Summary:**\n\n{df.describe().to_string()}")
            pdf.ln(5)

            # Save PDF
            report_file = io.BytesIO()
            pdf.output(report_file)
            report_file.seek(0)
            st.download_button("📥 Download Report", report_file, file_name="Data_Report.pdf", mime="application/pdf")

        # 📤 Export Cleaned Data
        st.write("### 📤 Export Cleaned Data")
        cleaned_file = io.BytesIO()
        df.to_csv(cleaned_file, index=False)
        cleaned_file.seek(0)
        st.download_button("Download Cleaned Data", cleaned_file, file_name="cleaned_data.csv", mime="text/csv")

    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")
