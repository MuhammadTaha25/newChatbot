from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
openai_client = wrap_openai(OpenAI())

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']
@traceable
def create_expert_chain(LLM=None, retriever=None):
    """
    Create a chain for answering questions as an expert on Elon Musk.

    Parameters:
        llm (object): The language model to use for generating responses.
        retriever (object): A retriever for fetching relevant context based on the question.

    Returns:
        object: A configured chain for answering questions about Elon Musk.
    """
    # Define the prompt template
    prompt_str ="""You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about Elon Musk.

Answer all questions as if you are an expert on his life, career, companies, and achievements.

IMPORTANT LANGUAGE INSTRUCTION:
You must always respond in Urdu language written in Roman Urdu. 
Your tone should be natural, conversational, and friendly like a native Urdu speaker.
Do not respond in English unless specifically asked.

If a user asks a question that is not related to Elon Musk or the provided context, respond with:
"Main sirf Elon Musk se related sawalat ka jawab dene ke liye train kiya gaya hoon."

Context: {context}
Current Question: {question}

Please provide a helpful response based on the context and chat history, and respond only in Urdu (Roman Urdu)."""
    
    _prompt = ChatPromptTemplate.from_template(prompt_str)

    # Chain setup with history
    setup = {
        "question": itemgetter("question"),
        "context": itemgetter("question") | retriever | format_docs,
    }
    _chain = setup | _prompt | LLM | StrOutputParser()

    return _chain
