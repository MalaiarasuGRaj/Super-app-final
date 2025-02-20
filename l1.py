import streamlit as st
import pandas as pd
import requests

API_BASE = "https://api.sambanova.ai/v1"
MODEL_405B = "Meta-Llama-3.3-70B-Instruct"
API_KEY = "f2321685-3794-4924-91dd-a0d9ee7c365b"

def extract_column_names(llm_response):
    location_col = llm_response.split('location_column:')[1].split(',')[0].strip()
    regional_col = llm_response.split('regional_column:')[1].strip()
    return location_col, regional_col

def analyze_columns_with_llm(df):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
   
    sample_data = df.head(5).to_string()
    columns_list = ", ".join(df.columns.tolist())
   
    prompt = f"""Given these column names: {columns_list}
    And sample data:
    {sample_data}
   
    Which column contains business location (country) information and which column contains regional information?
    Return only the exact column names in this format: 'location_column: <name>, regional_column: <name>'"""

    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert in data validation for region and location mismatches."},
            {"role": "user", "content": prompt}
        ],
        "model": MODEL_405B,
        "max_tokens": 4000,
        "temperature": 0.1,
        "top_p": 0.1,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1
    }

    try:
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except (requests.exceptions.RequestException, KeyError) as e:
        return f"Error analyzing columns: {str(e)}"

def check_region_location_mismatch(df, location_column, regional_column):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    You are an expert in data validation for region and location mismatches. You are given a dataset with two columns, one is business location and the other is regional. You have to compare the location and regional columns and identify whether the regional column is correct with respect to the location column.

    Your task is to identify ALL mismatches based on these **strict rules**:

    ### **Matching Rules:**
    1. If business location column contains **only EMEA countries**, then regional must only be 'EMEA'.
    2. If business location column contains **only APAC countries**, then regional must only be 'APAC'.
    3. If business location column contains **both APAC and EMEA countries**, then regional must be 'APAC/EMEA'.
    4. If business location is Worldwide or contains multiple regions, then regional **must** be Global.
    5. If regional contains 'APAC/EMEA' but the location is only APAC or EMEA, then it is a mismatch.
    6. If location contains both countries of "APAC and EMEA" but the location is only 'APAC' or 'EMEA', then it is a mismatch.
    7. The location column should not have any business locations like 'EMEA' or 'APAC'.
    8. If location is worldwide, regional must be global (Not "Global/APAC" and "Global/EMEA").
    9. You **must not** consider blanks in regional as a mismatch, only if location and regional is filled then take it for the comparision, if regional is empty then skip the comparision.
    10. If location is worldwide and regional is global then it is **not a mismatch**.

    Output Format:
    Regional : [Row Index]
    **Only display the tables, no additional text.**
    """

    df[location_column] = df[location_column].str.strip().str.upper()
    df[regional_column] = df[regional_column].str.strip().str.upper()

    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert in data validation for region and location mismatches."},
            {"role": "user", "content": f"{prompt}\n\nData:\n{df[[location_column, regional_column]].to_string()}"}
        ],
        "model": MODEL_405B,
        "max_tokens": 4000,
        "temperature": 0.1,
        "top_p": 0.1,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1
    }
    try:
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except (requests.exceptions.RequestException, KeyError) as e:
        return None
def main():
    st.title("File Upload & Location Analysis")
   
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
   
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
       
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
           
        st.dataframe(df)
       
        llm_result = analyze_columns_with_llm(df)
        location_column, regional_column = extract_column_names(llm_result)
       
        st.write(f"Location Column: {location_column}")
        st.write(f"Regional Column: {regional_column}")
       
        mismatches = check_region_location_mismatch(df, location_column, regional_column)
        if mismatches:
            st.write("Mismatches Found:")
            st.write(mismatches)
if __name__ == "__main__":
    main()