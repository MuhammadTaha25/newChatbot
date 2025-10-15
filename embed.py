from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
def initialize_embeddings(openai_api_key=OPENAI_API_KEY):
    """
    Initialize embeddings using OpenAI or HuggingFace based on the availability of the OpenAI API key.

    Parameters:
        openai_api_key (str, optional): Your OpenAI API key. If not provided, it checks the environment variable.

    Returns:
        Embeddings object: An instance of OpenAIEmbeddings or HuggingFaceEmbeddings.
    """
    # Retrieve the OpenAI API key (default to an environment variable if not explicitly provided)
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    # if openai_api_key:  # Use OpenAI embeddings if the API key is available
    #     embeddings = OpenAIEmbeddings(
    #         model="text-embedding-3-small",  # Use the desired OpenAI model
    #         openai_api_key=openai_api_key
    #     )
    #     print("Using OpenAIEmbeddings")
    # else:  # Fallback to HuggingFace embeddings if no OpenAI API key is found
    embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"  # Use the desired HuggingFace model
        )
        print("Using HuggingFaceEmbeddings")
    
    return embeddings
