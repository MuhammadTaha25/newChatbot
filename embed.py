from langchain_openai import OpenAIEmbeddings  # Importing OpenAI embeddings
from langchain_huggingface import HuggingFaceEmbeddings  # Importing HuggingFace embeddings
import streamlit as st  # Importing Streamlit for secret management

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Fetching the OpenAI API key from Streamlit secrets

def initialize_embeddings(openai_api_key=OPENAI_API_KEY):
    """
    Initialize embeddings using OpenAI or HuggingFace based on the availability of the OpenAI API key.

    Parameters:
        openai_api_key (str, optional): Your OpenAI API key. If not provided, it checks the environment variable.

    Returns:
        Embeddings object: An instance of OpenAIEmbeddings or HuggingFaceEmbeddings.
    """
    # Check if OpenAI API key is provided or retrieve from environment variable
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if openai_api_key:  # If OpenAI API key is available, use OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Define the OpenAI model
            openai_api_key=openai_api_key
        )
        print("Using OpenAIEmbeddings")  # Log which embeddings are being used
    else:  # If no OpenAI API key, fallback to HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"  # Define the HuggingFace model
        )
        print("Using HuggingFaceEmbeddings")  # Log which embeddings are being used
    
    return embeddings  # Return the initialized embeddings object
