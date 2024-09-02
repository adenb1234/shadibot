import streamlit as st
import anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load the context from the text file
@st.cache_resource
def load_context():
    with open("columnist_context.txt", "r", encoding="utf-8") as f:
        return f.read()

context = load_context()

# Streamlit UI
st.title("Columnist Bot")

# User input
user_input = st.text_input("Ask a question about the columnist's writings:")

if user_input:
    # Prepare the message for Claude
    message = f"""
    You are an AI assistant trained on a collection of columns written by a columnist.
    Use the following context to answer the user's question:

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

# Instructions for running the app
st.sidebar.header("How to use")
st.sidebar.write("""
1. Ensure you have run `generate_context.py` to create the 'columnist_context.txt' file.
2. Make sure you have the required libraries installed:
   ```
   pip install streamlit anthropic python-dotenv
   ```
3. Create a .env file in the same directory as this script and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
""")
