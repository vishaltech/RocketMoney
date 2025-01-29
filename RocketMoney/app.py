import streamlit as st
import pandas as pd
import yaml
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# Initialize OpenAI API for NLP analysis
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI key
llm = OpenAI(api_token=OPENAI_API_KEY)
pandas_ai = PandasAI(llm)

def analyze_data(df):
    """Generate insights from the uploaded file using AI."""
    try:
        insights = pandas_ai.run(df, prompt="Provide a summary of key insights from this dataset.")
        return insights
    except Exception as e:
        return f"Error generating insights: {e}"

def main():
    st.title("📊 File Upload & AI-Powered Insights")

    # File uploader
    uploaded_file = st.file_uploader("📂 Upload a file", type=['xls', 'xlsx', 'csv'])

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
                st.error("🚫 Unsupported file type.")
                return

            st.success("✅ File uploaded successfully!")
            st.dataframe(df)  # Display the dataframe

            # Generate AI-powered insights
            st.subheader("🔍 AI-Generated Insights")
            insights = analyze_data(df)
            st.write(insights)

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

if __name__ == "__main__":
    main()
