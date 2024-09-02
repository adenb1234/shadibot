import streamlit as st
import pandas as pd
import openai
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def prepare_data_for_finetuning():
    # Load the CSV data
    df = pd.read_csv("columnist_data.csv")
    
    # Prepare the data in the format OpenAI expects
    fine_tuning_data = []
    for _, row in df.iterrows():
        prompt = f"Write a column in the style of Shadi Hamid on the topic: {row['Title']}"
        completion = f"{row['Text']}\n\nEND OF TEXT"
        fine_tuning_data.append({"prompt": prompt, "completion": completion})
    
    # Save the prepared data
    with open("fine_tuning_data.jsonl", "w") as f:
        for item in fine_tuning_data:
            f.write(json.dumps(item) + "\n")
    
    print("Data prepared for fine-tuning.")

def fine_tune_model():
    # Upload the file
    with open("fine_tuning_data.jsonl", "rb") as f:
        response = openai.File.create(file=f, purpose='fine-tune')
    file_id = response.id

    # Create a fine-tuning job
    response = openai.FineTuningJob.create(
        training_file=file_id, 
        model="gpt-3.5-turbo"
    )
    job_id = response.id

    print(f"Fine-tuning job created with ID: {job_id}")
    print("Please wait for the job to complete. You can check the status using OpenAI's API or dashboard.")
    return job_id

def check_fine_tuning_status(job_id):
    response = openai.FineTuningJob.retrieve(job_id)
    return response.status

def run_columnist_bot():
    st.title("Shadi Hamid Columnist Bot")

    # Check if we have a fine-tuned model ID saved
    if 'fine_tuned_model_id' not in st.session_state:
        st.session_state.fine_tuned_model_id = st.text_input("Enter your fine-tuned model ID:")

    if st.session_state.fine_tuned_model_id:
        # User input
        user_input = st.text_input("Provide a topic for Shadi Hamid to write about:")

        if user_input:
            # Prepare the message for the fine-tuned model
            prompt = f"Write a column in the style of Shadi Hamid on the topic: {user_input}"

            # Call OpenAI API with the fine-tuned model
            try:
                response = openai.ChatCompletion.create(
                    model=st.session_state.fine_tuned_model_id,
                    messages=[
                        {"role": "system", "content": "You are an AI writing assistant trained on a collection of columns written by a columnist named Shadi Hamid. You are meant to mimic his tone, his style, and his vocabulary as closely as possible. Do not rely on other, outside writing texts except for factual information. Use only his columns to write in his voice and style."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )

                # Display the model's response
                st.write("Shadi Hamid Bot:", response.choices[0].message['content'])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a fine-tuned model ID to use the Shadi Hamid Bot.")

    # Instructions for running the app
    st.sidebar.header("How to use")
    st.sidebar.write("""
    1. Ensure you have run the fine-tuning process with Shadi Hamid's columns and have a fine-tuned model ready.
    2. Enter your fine-tuned model ID when prompted.
    3. Provide topics for the bot to write about in Shadi Hamid's style.
    """)

if __name__ == "__main__":
    # Check if we're in Streamlit
    if st.runtime.exists():
        run_columnist_bot()
    else:
        # If not in Streamlit, prepare data and start fine-tuning
        prepare_data_for_finetuning()
        job_id = fine_tune_model()
        print(f"Initial status: {check_fine_tuning_status(job_id)}")
        print("Run this script with Streamlit once fine-tuning is complete.")
