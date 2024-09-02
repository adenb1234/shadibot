import streamlit as st
import pandas as pd
import anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Function to generate context from CSV
def generate_context_from_csv():
    df = pd.read_csv("columnist_data.csv")
    context = "\n\n".join([
        f"Title: {row['Title']}\nDate: {row['Date']}\nText: {row['Text']}"
        for _, row in df.iterrows()
    ])
    return context

# Load or generate the context
@st.cache_resource
def load_or_generate_context():
    context_file = "columnist_context.txt"
    if os.path.exists(context_file):
        with open(context_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        context = generate_context_from_csv()
        with open(context_file, "w", encoding="utf-8") as f:
            f.write(context)
        return context

context = load_or_generate_context()

# Streamlit UI
st.title("Columnist Bot")

# User input
user_input = st.text_input("Ask a question about the columnist's writings:")

if user_input:
    # Prepare the message for Claude
    message = f"""
    You are an AI writing assistant trained on a collection of columns written by a columnist named Shadi Hamid. You are meant to mimic his tone, his style, and his vocabulary as closely as possible. Do not rely on other, outside writing texts except for factual information. 
    Use only his columns to answer the user's questions about writing in his voice and style:

    {context}

    User question: {user_input}

    Please provide a concise and relevant answer based on the columnist's writings.
    """

    # Call Claude API
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.5,
        messages=[
            {"role": "user", "content": message}
        ]
    )

    # Display Claude's response
    st.write("Columnist Bot:", response.content[0].text)

