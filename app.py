LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="muskchatbot"
from pineconedb import manage_pinecone_store
from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM
import streamlit as st
from streamlit_mic_recorder import speech_to_text
OPENAI_API_KEY =st.secrets['OPENAI_API_KEY']
GOOGLE_API_KEY =st.secrets['google_api_key']

# app.py
from dotenv import load_dotenv
import os

load_dotenv()   # ← loads everything from .env into os.environ

import streamlit as st
from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM
from streamlit_mic_recorder import speech_to_text

# now fetch them exactly once
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME")
# And LangSmith
LANGSMITH_TRACING  = os.getenv("LANGSMITH_TRACING") == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT")

# pass those into initialize_LLM (you might need to extend its signature)
LLM = initialize_LLM(openai_api_key=OPENAI_API_KEY,
                     gemini_api_key=GOOGLE_API_KEY,
                     # langsmith_api_key=LANGSMITH_API_KEY,
                     # langsmith_endpoint=LANGSMITH_ENDPOINT,
                     # langsmith_project=LANGSMITH_PROJECT,
                     # langsmith_tracing=LANGSMITH_TRACING)

retriever = manage_pinecone_store(index_name=PINECONE_INDEX, embeddings=...)
chain     = create_expert_chain(LLM, retriever)

st.title("Ask anything about Musk")
…
retriever=manage_pinecone_store()
chain=create_expert_chain(LLM,retriever)
# Build the chain
# Set the title of the app
# Initialize components
# Chat container to display conversation
st.title("Ask anything about Musk")
chat_container = st.container()
if "messages" not in st.session_state:
    st.session_state.messages = []
def send_input():
    st.session_state.send_input=True
   
query= st.text_input("Please enter a query", key="query", on_change=send_input)
send_button = st.button("Send", key="send_btn")  # Single send button
    
voice_recording=speech_to_text(language="en",use_container_width=True,just_once=True,key="STT")
    
if voice_recording:
    query=voice_recording
    
# Chat logic
if query or voice_recording:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response =chain.stream({'question': query})
        print(response)

    # Generate response
    # Update session state with user query and AI response
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))
    
with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message) 
