from langchain.prompts.chat import ChatPromptTemplate
import streamlit as st
from langchain.schema import StrOutputParser
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
    prompt_str = """
"You are a helpful and knowledgeable customer support chatbot for an ecommerce store. You are an expert on all topics related to online shopping, including products, orders, shipping, returns, payments, and website help.
Your most important rule: If a user asks a question that is completely unrelated to ecommerce, you must respond with this exact phrase: 'I am trained to answer ecommerce-related queries only'

Chat History:
{history}

Context: {context}
Current Question: {question}

Please provide a helpful response based on the context and chat history."""
    
    _prompt = ChatPromptTemplate.from_template(prompt_str)

    # Chain setup with history
    setup = {
        "question": itemgetter("question"),
        "context": itemgetter("question") | retriever | format_docs,
        "history": itemgetter("history")
    }
    _chain = setup | _prompt | LLM | StrOutputParser()

    return _chain

def format_chat_history(history_list):
    """Format chat history for the prompt"""
    if not history_list:
        return "No previous conversation."
    
    formatted_history = ""
    for message in history_list:
        if message["role"] == "user":
            formatted_history += f"User: {message['content']}\n"
        else:
            formatted_history += f"Assistant: {message['content']}\n"
    
    return formatted_history

def get_chat_response(question, llm, retriever):
    """Get response from LLM with chat history"""
    # Format history for the prompt
    formatted_history = format_chat_history(st.session_state.chat_history)
    
    # Create chain
    chain = create_expert_chain(LLM=llm, retriever=retriever)
    
    # Get response
    _chain = chain.invoke({
        "question": question,
        "history": formatted_history
    })
    
    # Update chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    return _chain
