import streamlit as st
import pandas as pd
import yaml
from pandasai import SmartDataframe  # Correct import
from pandasai.llm.openai import OpenAI

# OpenAI API Key Setup
OPENAI_API_KEY = "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"

# Initialize OpenAI API for NLP analysis
llm = OpenAI(api_token=OPENAI_API_KEY)

def main():
    st.title("🚀 Rocket Money Dashboard")
    st.write("## AI-Powered Data Analysis with PandasAI")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])  
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        sdf = SmartDataframe(df, config={"llm": llm})
        
        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())
        
        query = st.text_input("Ask a question about the data:")
        if st.button("Analyze Data"):
            with st.spinner("Processing..."):
                response = sdf.chat(query)
                st.write("### AI Response:")
                st.write(response)

if __name__ == "__main__":
    main()
