import streamlit as st
import pandas as pd
import plotly.express as px
import io
import numpy as np

# Streamlit UI Configuration
st.set_page_config(page_title="Ultimate Data Analyzer", layout="wide")
st.title("📊 Ultimate Data Analyzer")
st.write("Upload an Excel or CSV file and explore data with powerful tools and visualizations!")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Display dataframe preview
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Data Cleaning & Processing
    st.write("### 🛠 Data Cleaning & Processing")
    
    if st.button("Drop Duplicates"):
        df.drop_duplicates(inplace=True)
        st.write("Duplicates removed successfully!")
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        if st.button("Fill Missing Values with Mean"):
            df.fillna(df.mean(), inplace=True)
            st.write("Missing values filled with column mean!")
    
    # Column Selection
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Data Summary
    st.write("### 📌 Data Summary")
    st.write(df.describe())
    
    # Correlation Heatmap
    st.write("### 🔥 Correlation Heatmap")
    if numeric_columns:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    # Interactive Visualization
    st.write("### 📊 Interactive Visualization")
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Pie Chart"])
    x_axis = st.selectbox("Select X-axis", numeric_columns + categorical_columns)
    y_axis = st.selectbox("Select Y-axis", numeric_columns)
    
    if st.button("Generate Chart"):
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart of {y_axis} vs {x_axis}")
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot of {y_axis} vs {x_axis}")
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"Line Chart of {y_axis} vs {x_axis}")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_axis, title=f"Pie Chart of {x_axis}")
        else:
            fig = px.histogram(df, x=x_axis, title=f"Histogram of {x_axis}")
        
        st.plotly_chart(fig)
    
    # Data Export
    st.write("### 📤 Export Cleaned Data")
    cleaned_file = io.BytesIO()
    df.to_csv(cleaned_file, index=False)
    cleaned_file.seek(0)
    st.download_button("Download Cleaned Data", cleaned_file, file_name="cleaned_data.csv", mime="text/csv")
