import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get the fine-tuned model ID from environment variable
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL_ID")

if not FINE_TUNED_MODEL:
    st.error("Fine-tuned model ID is not set. Please set the FINE_TUNED_MODEL_ID environment variable.")
    st.stop()

def get_ai_response(prompt, max_tokens=500):
    try:
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI writing assistant trained to mimic the style, tone, and vocabulary of columnist Shadi Hamid. You can write, edit, and draft in his voice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    st.title("Shadi Hamid Writing Assistant")

    st.sidebar.title("Choose a Task")
    task = st.sidebar.radio("Select a task:", ["Write", "Edit", "Draft Outline"])

    if task == "Write":
        st.header("Write in Shadi Hamid's Style")
        topic = st.text_input("Enter a topic:")
        if topic:
            prompt = f"Write a column in the style of Shadi Hamid on the topic: {topic}"
            response = get_ai_response(prompt, max_tokens=1000)
            st.write(response)

    elif task == "Edit":
        st.header("Edit Text in Shadi Hamid's Style")
        text_to_edit = st.text_area("Enter the text you want to edit:")
        if text_to_edit:
            prompt = f"Edit the following text to match Shadi Hamid's writing style: {text_to_edit}"
            response = get_ai_response(prompt, max_tokens=1000)
            st.write(response)

    elif task == "Draft Outline":
        st.header("Draft Outline in Shadi Hamid's Style")
        topic = st.text_input("Enter a topic for the outline:")
        if topic:
            prompt = f"Create an outline for a column by Shadi Hamid on the topic: {topic}"
            response = get_ai_response(prompt, max_tokens=500)
            st.write(response)

if __name__ == "__main__":
    main()
