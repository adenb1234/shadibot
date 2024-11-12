import streamlit as st
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize OpenAI client
def init_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    return OpenAI(api_key=api_key)

client = init_openai_client()

# Initialize BERT model
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load and process articles
@st.cache_data
def load_articles():
    try:
        # Load CSV from GitHub URL
        github_url = "https://raw.githubusercontent.com/shadihamid/columnist_data/main/columnist_data.csv"
        df = pd.read_csv(github_url)
        
        # Combine title and text for better context
        df['full_text'] = df.apply(lambda row: f"Title: {row['title']}\n\nContent: {row['text']}", axis=1)
        
        # Get embeddings
        model = load_bert_model()
        embeddings = model.encode(df['full_text'].tolist(), show_progress_bar=True)
        
        return df, embeddings
    except Exception as e:
        st.error(f"Error loading articles: {str(e)}")
        return None, None

def get_relevant_articles(query, df, embeddings, n_results=5):
    """Get most relevant articles based on semantic search"""
    try:
        model = load_bert_model()
        query_embedding = model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top n_results
        top_indices = np.argsort(similarities)[-n_results:][::-1]
        
        relevant_articles = df.iloc[top_indices]['full_text'].tolist()
        relevant_metadata = df.iloc[top_indices][['title', 'date']].to_dict('records')
        
        return relevant_articles, relevant_metadata
    except Exception as e:
        st.error(f"Error in semantic search: {str(e)}")
        return [], []

def get_ai_response(prompt, relevant_articles, relevant_metadata, max_tokens=1000):
    """Generate response using GPT-4 with context from relevant articles"""
    try:
        # Create context from relevant articles
        context = "Here are some relevant examples of the columnist's writing to inform your response style:\n\n"
        for i, (article, metadata) in enumerate(zip(relevant_articles, relevant_metadata), 1):
            context += f"Example {i} (Title: {metadata['title']}, Date: {metadata['date']}):\n{article}\n\n"
        
        system_prompt = (
            "You are an AI writing assistant trained to mimic the style, tone, and vocabulary "
            "of the columnist based on the provided examples. Pay attention to their typical "
            "sentence structure, argument patterns, word choice, and how they develop their ideas. "
            "Your goal is to write new content that could believably have been written by them."
        )
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Columnist Writing Assistant",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("Columnist Writing Assistant")
    
    # Load articles and embeddings
    df, embeddings = load_articles()
    if df is None or embeddings is None:
        st.error("Failed to load articles database")
        st.stop()
    
    st.markdown("""
    This app helps you generate content in the columnist's style using semantic search 
    to find relevant examples from their previous columns.
    """)
    
    st.sidebar.title("Choose a Task")
    task = st.sidebar.radio(
        "Select a task:",
        ["Write", "Edit", "Draft Outline"]
    )
    
    # Debug option to show relevant articles
    show_relevant = st.sidebar.checkbox("Show relevant articles used for context")
    
    if task == "Write":
        st.header("Write in Columnist's Style")
        topic = st.text_input("Enter a topic:")
        if st.button("Generate") and topic:
            with st.spinner("Finding relevant articles and generating content..."):
                # Get relevant articles
                relevant_articles, relevant_metadata = get_relevant_articles(topic, df, embeddings)
                
                if show_relevant:
                    st.subheader("Relevant Articles Used for Context")
                    for i, (article, metadata) in enumerate(zip(relevant_articles, relevant_metadata), 1):
                        with st.expander(f"Article {i} - {metadata['title']} ({metadata['date']})"):
                            st.write(article)
                
                prompt = f"Write a column in the columnist's style on the topic: {topic}"
                response = get_ai_response(prompt, relevant_articles, relevant_metadata)
                if response:
                    st.markdown("### Generated Column")
                    st.markdown(response)
    
    elif task == "Edit":
        st.header("Edit Text in Columnist's Style")
        text_to_edit = st.text_area("Enter the text you want to edit:", height=200)
        if st.button("Edit") and text_to_edit:
            with st.spinner("Finding relevant articles and editing..."):
                relevant_articles, relevant_metadata = get_relevant_articles(text_to_edit, df, embeddings)
                
                if show_relevant:
                    st.subheader("Relevant Articles Used for Context")
                    for i, (article, metadata) in enumerate(zip(relevant_articles, relevant_metadata), 1):
                        with st.expander(f"Article {i} - {metadata['title']} ({metadata['date']})"):
                            st.write(article)
                
                prompt = f"Edit the following text to match the columnist's writing style: {text_to_edit}"
                response = get_ai_response(prompt, relevant_articles, relevant_metadata)
                if response:
                    st.markdown("### Edited Version")
                    st.markdown(response)
    
    elif task == "Draft Outline":
        st.header("Draft Outline in Columnist's Style")
        topic = st.text_input("Enter a topic for the outline:")
        if st.button("Generate Outline") and topic:
            with st.spinner("Finding relevant articles and generating outline..."):
                relevant_articles, relevant_metadata = get_relevant_articles(topic, df, embeddings)
                
                if show_relevant:
                    st.subheader("Relevant Articles Used for Context")
                    for i, (article, metadata) in enumerate(zip(relevant_articles, relevant_metadata), 1):
                        with st.expander(f"Article {i} - {metadata['title']} ({metadata['date']})"):
                            st.write(article)
                
                prompt = f"Create an outline for a column on the topic: {topic}. Structure it similar to how the columnist typically organizes their arguments and ideas."
                response = get_ai_response(prompt, relevant_articles, relevant_metadata)
                if response:
                    st.markdown("### Generated Outline")
                    st.markdown(response)

if __name__ == "__main__":
    main()
