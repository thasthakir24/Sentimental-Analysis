import os
import json
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("Sentiment Analysis (CSV Upload)")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def analyze_sentiment(text):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
You are a sentiment analysis assistant.

Analyze each sentence separately and return output strictly in JSON format like this:

{
  "Positive": ["sentence1", "sentence2"],
  "Negative": ["sentence1"],
  "Neutral": ["sentence1"]
}

Do not return anything except valid JSON.
"""
            },
            {
                "role": "user",
                "content": f"Analyze this paragraph:\n{text}"
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # 🔴 CHANGE THIS COLUMN NAME IF NEEDED
    text_column = st.selectbox(
        "Select the column containing text",
        df.columns
    )

    if st.button("Analyze Sentiment"):
        all_results = []

        for idx, text in enumerate(df[text_column]):
            if pd.isna(text):
                continue

            result = analyze_sentiment(str(text))

            try:
                json_result = json.loads(result)
                all_results.append({
                    "row": idx,
                    "sentiment": json_result
                })
            except:
                all_results.append({
                    "row": idx,
                    "error": "Invalid JSON returned",
                    "raw_output": result
                })

        st.success("Analysis complete!")
        st.json(all_results)
