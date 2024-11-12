# app.py
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with error handling
def init_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    return OpenAI(api_key=api_key)

# Initialize OpenAI client
client = init_openai_client()

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
                {
                    "role": "system", 
                    "content": "You are an AI writing assistant trained to mimic the style, tone, and vocabulary of columnist Shadi Hamid. You can write, edit, and draft in his voice."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while getting AI response: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Shadi Hamid Writing Assistant",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("Shadi Hamid Writing Assistant")
    
    # Add a brief description
    st.markdown("""
    This app helps you generate content in the style of columnist Shadi Hamid.
    Choose a task from the sidebar to get started.
    """)
    
    st.sidebar.title("Choose a Task")
    task = st.sidebar.radio(
        "Select a task:",
        ["Write", "Edit", "Draft Outline"],
        key="task_selection"
    )
    
    if task == "Write":
        st.header("Write in Shadi Hamid's Style")
        topic = st.text_input("Enter a topic:", key="write_topic")
        if st.button("Generate", key="write_button"):
            if topic:
                with st.spinner("Generating content..."):
                    prompt = f"Write a column in the style of Shadi Hamid on the topic: {topic}"
                    response = get_ai_response(prompt, max_tokens=1000)
                    if response:
                        st.markdown(response)
            else:
                st.warning("Please enter a topic first.")
    
    elif task == "Edit":
        st.header("Edit Text in Shadi Hamid's Style")
        text_to_edit = st.text_area(
            "Enter the text you want to edit:",
            height=200,
            key="edit_text"
        )
        if st.button("Edit", key="edit_button"):
            if text_to_edit:
                with st.spinner("Editing content..."):
                    prompt = f"Edit the following text to match Shadi Hamid's writing style: {text_to_edit}"
                    response = get_ai_response(prompt, max_tokens=1000)
                    if response:
                        st.markdown("### Edited Version")
                        st.markdown(response)
            else:
                st.warning("Please enter some text to edit.")
    
    elif task == "Draft Outline":
        st.header("Draft Outline in Shadi Hamid's Style")
        topic = st.text_input("Enter a topic for the outline:", key="outline_topic")
        if st.button("Generate Outline", key="outline_button"):
            if topic:
                with st.spinner("Generating outline..."):
                    prompt = f"Create an outline for a column by Shadi Hamid on the topic: {topic}"
                    response = get_ai_response(prompt, max_tokens=500)
                    if response:
                        st.markdown(response)
            else:
                st.warning("Please enter a topic first.")

if __name__ == "__main__":
    main()
