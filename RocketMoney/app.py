import streamlit as st
import pandas as pd
import yaml
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import openai

# Load OpenAI API key
st.set_page_config(page_title="AI-Powered File Analysis", layout="wide")

def main():
    st.title("📂 AI-Powered File Analyzer")

    # File uploader
    uploaded_file = st.file_uploader("📂 Upload a file (CSV, XLSX, XLS)", type=['xls', 'xlsx', 'csv'])

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
            st.dataframe(df.head(10))  # Show first 10 rows
            
            # Generate AI-powered insights
            generate_insights(df)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

def generate_insights(df):
    """ Use AI to analyze and summarize key insights from the dataset """
    st.write("### 📊 AI-Generated Key Insights")

    try:
        # Initialize PandasAI with OpenAI LLM
        openai_api_key = "your-openai-api-key-here"  # Replace with your actual OpenAI key
        llm = OpenAI(api_token=openai_api_key)
        smart_df = SmartDataframe(df, config={"llm": llm})

        # AI-generated analysis
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
