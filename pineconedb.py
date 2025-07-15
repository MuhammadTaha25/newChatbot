from langchain_pinecone import PineconeVectorStore
from chunked_doc import chunking_documents
from embed import initialize_embeddings
from dotenv import load_dotenv
import os
import streamlit as st
# Load environment variables

# Get Pinecone index name from environment
PINECONE_INDEX = st.secrets["PINECONE_INDEX_NAME"]
os.environ['PINECONE_API_KEY']=st.secrets['PINECONE_API_KEY']
# Initialize embeddings only once
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
    if not index_name:
        raise ValueError("Pinecone index name (PINECONE_INDEX_NAME) is not set in the environment.")

    try:
        # Attempt to load an existing Pinecone index
        pineconedb = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        retriever = pineconedb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        print(f"Successfully loaded existing Pinecone index: {index_name}")
        return retriever
    except Exception as e:
        print(f"Error while loading Pinecone index: {e}")
        print(f"Attempting to create a new Pinecone index: {index_name}")

        # Retrieve chunked documents
        chunks_received = chunking_documents()
        if not chunks_received:
            raise ValueError("No documents returned by chunking_documents.")

        # Create a new Pinecone vector store with the processed chunks
        pineconedb = PineconeVectorStore.from_documents(
            chunks_received,
            embeddings,
            index_name=index_name
        )
        retriever = pineconedb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        print(f"New Pinecone index created: {index_name}")
        return retriever 
