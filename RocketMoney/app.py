import streamlit as st
import pandas as pd
import yaml
import os
from pandasai.smart_dataframe import SmartDataframe
from pandasai.llm.openai import OpenAI
import openai

# Set OpenAI API key (Ensure it's stored securely)
os.environ["OPENAI_API_KEY"] = "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"

st.set_page_config(page_title="🚀 AI-Powered Data Analysis", layout="wide")

def main():
    st.title("📊 AI-Powered File Analyzer")
    uploaded_file = st.file_uploader("📂 Upload a CSV/XLSX/XLS file", type=['xls', 'xlsx', 'csv'])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        try:
            if file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_extension == 'xls':
                df = pd.read_excel(uploaded_file, engine='xlrd')
            elif file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                st.error("❌ Unsupported file type.")
                return

            st.success("✅ File uploaded successfully!")
            st.write("### 🔍 Data Preview")
            st.dataframe(df.head(10))

            # AI Analysis
            generate_ai_insights(df)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

def generate_ai_insights(df):
    """ Use AI to analyze and summarize key insights from the dataset """
    st.write("### 📈 AI-Generated Key Insights")

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("❌ OpenAI API key is missing!")
            return

        llm = OpenAI(api_token=openai_api_key)
        smart_df = SmartDataframe(df, config={"llm": llm})

        questions = [
            "What are the key trends in this dataset?",
            "What are the most important columns?",
            "Is there any missing data?",
            "Can you summarize the top insights in one paragraph?"
        ]

        for question in questions:
            response = smart_df.chat(question)
            st.write(f"**🔹 {question}**")
            st.write(response)

    except Exception as e:
        st.error(f"⚠️ AI Analysis Failed: {e}")

if __name__ == "__main__":
    main()
