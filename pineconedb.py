from langchain_pinecone import PineconeVectorStore  # Importing Pinecone vector store module
from chunked_doc import chunking_documents  # Importing chunking_documents function to process content
from embed import initialize_embeddings  # Importing the initialize_embeddings function for embeddings
from dotenv import load_dotenv  # Importing dotenv for environment variable management
import os  # Importing OS for environment variable handling
import streamlit as st  # Importing Streamlit for secret management

# Load environment variables

# Get Pinecone index name from environment (Streamlit secrets)
PINECONE_INDEX = st.secrets["PINECONE_INDEX_NAME"]

# Initialize embeddings only once for reuse
embeddings = initialize_embeddings()

def manage_pinecone_store(index_name=PINECONE_INDEX, embeddings=embeddings):
    """
    Manage Pinecone vector store by checking for an existing index or creating a new one.

    Parameters:
        index_name (str): The name of the Pinecone index.
        embeddings (object): Embedding model used for generating vector representations.

    Returns:
        retriever (object): The retriever for fetching relevant chunks of data.
    """
    if not index_name:  # Ensure index name is provided
        raise ValueError("Pinecone index name (PINECONE_INDEX_NAME) is not set in the environment.")

    try:
        # Attempt to load the existing Pinecone index
        pineconedb = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        retriever = pineconedb.as_retriever(search_type="mmr", search_kwargs={"k": 5})  # Set up retriever
        print(f"Successfully loaded existing Pinecone index: {index_name}")
        return retriever
    except Exception as e:
        # Handle errors when loading the index, and attempt to create a new one
        print(f"Error while loading Pinecone index: {e}")
        print(f"Attempting to create a new Pinecone index: {index_name}")

        # Retrieve chunked documents to be indexed
        chunks_received = chunking_documents()
        if not chunks_received:  # Ensure chunks are received
            raise ValueError("No documents returned by chunking_documents.")

        # Create a new Pinecone vector store using the chunked documents
        pineconedb = PineconeVectorStore.from_documents(
            chunks_received,
            embeddings,
            index_name=index_name
        )
        retriever = pineconedb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        print(f"New Pinecone index created: {index_name}")
        return retriever

