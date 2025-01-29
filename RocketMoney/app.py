import streamlit as st
import pandas as pd
import openai
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# Set OpenAI API key
OPENAI_API_KEY = "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"
llm = OpenAI(api_token=OPENAI_API_KEY)

# Streamlit UI
st.title("🚀 AI-Powered Excel & CSV Analyzer")
st.write("Upload an Excel or CSV file and ask AI-powered questions about your data!")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Determine file type and read it
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Convert to SmartDataframe for AI-powered queries
    sdf = SmartDataframe(df, config={"llm": llm})
    
    st.write("### Ask Questions About Your Data")
    query = st.text_input("Type your question below")
    if st.button("Ask AI") and query:
        response = sdf.chat(query)
        st.write("**AI Answer:**", response)
