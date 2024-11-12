# app.py
import streamlit as st
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from numpy.linalg import norm

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

# Get embedding for a single text using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text.replace("\n", " ")  # Replace newlines with spaces for better embedding
    )
    return response.data[0].embedding

# Calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Load and process articles
@st.cache_data
def load_articles():
    try:
        # Load local CSV file
        df = pd.read_csv('columnist_data.csv')
        
        # Clean up the text data
        df['text'] = df['text'].fillna('')  # Replace NaN with empty string
        df['title'] = df['title'].fillna('Untitled')
        df['date'] = df['date'].fillna('Unknown date')
        
        # Combine title and text for better context
        df['full_text'] = df.apply(lambda row: f"Title: {row['title']}\n\nContent: {row['text']}", axis=1)
        
        # Get embeddings for all articles
        st.info("Generating embeddings for articles... This may take a few minutes.")
        embeddings = []
        for text in df['full_text']:
            embedding = get_embedding(text)
            embeddings.append(embedding)
        
        return df, embeddings
        
    except FileNotFoundError:
        st.error("Could not find columnist_data.csv file. Please make sure it's in the same directory as the script.")
        return None, None
    except Exception as e:
        st.error(f"Error loading articles: {str(e)}")
        st.write("Columns in CSV:", df.columns.tolist())  # Debug info
        return None, None

[rest of the code remains the same as in previous version...]
