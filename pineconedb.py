from langchain_pinecone import PineconeVectorStore
from chunked_doc import chunking_documents
from embed import initialize_embeddings
from dotenv import load_dotenv
import openai
import os

load_dotenv()

PINECONE_INDEX=os.getenv("PINECONE_INDEX_NAME")

embeddings=initialize_embeddings()
chunkDoc=chunking_documents()
def manage_pinecone_store(index_name=PINECONE_INDEX, embeddings=None):
    """
    Manage Pinecone vector store by checking for an existing index or creating a new one.

    Parameters:
        index_name (str): The name of the Pinecone index.
        url (str): The URL of the content source (e.g., a Wikipedia page).
        class_name (str): The class name to parse specific sections from the content.
        chunk_size (int, optional): Maximum character length of each chunk (default is 1500).
        chunk_overlap (int, optional): Overlapping characters between chunks for continuity (default is 100).
        embeddings (object, optional): Embedding model used for generating vector representations.
            Must be initialized before calling this function.

    Returns:
        PineconeVectorStore: The vector store created or loaded from Pinecone.
    """
    embeddings=initialize_embeddings()

    try:
        # Attempt to load an existing Pinecone index
        pineconedb = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        retriever=pineconedb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        print(f"Successfully loaded existing Pinecone index: {index_name}")
        return retriever
    except Exception as e:
        print(f"Error while loading Pinecone index: {e}")
        print(f"Attempting to create a new Pinecone index: {index_name}")

        # receiving chunked data
        chunks_received = chunking_documents()
        # Create a new vector store with the processed chunks
        pineconedb = PineconeVectorStore.from_documents(
            chunks_received,                # List of Document objects
            embeddings,            # Embedding model for generating vector representations
            index_name=index_name  # Name of the new Pinecone index
        )
        retriever=pineconedb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        print(f"New Pinecone index created: {index_name}")
        #retreiver is used to get the reevant chunks. 
        #can use mmr,similarity search , contextual compression retriver, etc
       
        return retriever
