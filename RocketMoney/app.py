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
