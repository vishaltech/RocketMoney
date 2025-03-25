import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Simple Data Analyzer", layout="wide")

st.title("Simple Data Analyzer")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Visualization")
    # Get numeric columns for charting
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        x_axis = st.selectbox("Select X axis", numeric_columns)
        y_axis = st.selectbox("Select Y axis", numeric_columns)
        chart_type = st.selectbox("Select chart type", ["Scatter", "Line", "Bar"])
        
        if st.button("Generate Chart"):
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis)
            elif chart_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis)
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No numeric columns available for visualization.")
