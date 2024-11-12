import streamlit as st
from openai import OpenAI
import chromadb
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from chromadb.utils import embedding_functions

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

# Initialize ChromaDB with OpenAI embeddings
@st.cache_resource
def init_chromadb():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-ada-002"
    )
    chroma_client = chromadb.Client()
    
    # Delete collection if it exists to avoid duplicates
    try:
        chroma_client.delete_collection("columnist_articles")
    except:
        pass
        
    collection = chroma_client.create_collection(
        name="columnist_articles",
        embedding_function=openai_ef
    )
    return collection

# Load and index articles
@st.cache_data
def load_and_index_articles():
    try:
        # Load CSV from GitHub URL
        github_url = "https://raw.githubusercontent.com/shadihamid/columnist_data/main/columnist_data.csv"
        df = pd.read_csv(github_url)
        
        # Get ChromaDB collection
        collection = init_chromadb()
        
        # Combine title and text for better context
        documents = [f"Title: {title}\n\nContent: {text}" 
                    for title, text in zip(df['title'], df['text'])]
        
        # Index each article
        collection.add(
            documents=documents,
            ids=[str(i) for i in range(len(df))],
            metadatas=[{
                "date": str(date),
                "title": str(title)
            } for date, title in zip(df['date'], df['title'])]
        )
        
        return True
    except Exception as e:
        st.error(f"Error loading and indexing articles: {str(e)}")
        return False

def get_relevant_articles(query, n_results=5):
    """Get most relevant articles based on semantic search"""
    collection = init_chromadb()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return results['documents'][0], results['metadatas'][0]  # Returns articles and their metadata

def get_ai_response(prompt, relevant_articles, relevant_metadata, max_tokens=1000):
    """Generate response using GPT-4 with context from relevant articles"""
    try:
        # Create context from relevant articles
        context = "Here are some relevant examples of the columnist's writing to inform your response style:\n\n"
        for i, (article, metadata) in enumerate(zip(relevant_articles, relevant_metadata), 1):
            # Include title and date in context
            context += f"Example {i} (Title: {metadata['title']}, Date: {metadata['date']}):\n{article}\n\n"
        
        # Create the complete prompt
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
    
    # Initialize database on first run
    if 'db_initialized' not in st.session_state:
        with st.spinner("Initializing database and loading articles..."):
            success = load_and_index_articles()
            if success:
                st.session_state.db_initialized = True
            else:
                st.error("Failed to initialize database")
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
                relevant_articles, relevant_metadata = get_relevant_articles(topic)
                
                if show_relevant:
                    st.subheader("Relevant Articles Used for Context")
                    for i, (article, metadata) in enumerate(zip(relevant_articles, relevant_metadata), 1):
                        with st.expander(f"Article {i} - {metadata['title']} ({metadata['date']})"):
                            st.write(article)
                
                # Generate response
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
                relevant_articles, relevant_metadata = get_relevant_articles(text_to_edit)
                
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
                relevant_articles, relevant_metadata = get_relevant_articles(topic)
                
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
