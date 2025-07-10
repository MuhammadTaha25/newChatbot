from pineconedb import manage_pinecone_store
from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

GOOGLE_API_KEY =st.secrets['google_api_key']
openai_client = wrap_openai(OpenAI())
LANGSMITH_TRACING='true'
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="muskchatbot"
OPENAI_API_KEY =st.secrets['OPENAI_API_KEY']

import streamlit as st
from streamlit_mic_recorder import speech_to_text
from creating_chain import create_expert_chain
from llModel import initialize_LLM
from pineconedb import manage_pinecone_store

def run_chatbot_app():
    st.title("Ask anything about Musk")
    
    # Initialize LLM and chain
    OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
    GOOGLE_API_KEY = st.secrets['google_api_key']
    LLM = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
    retriever = manage_pinecone_store()
    chain = create_expert_chain(LLM, retriever)

    chat_container = st.container()

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Define input send trigger
    def send_input():
        st.session_state.send_input = True

    # Text input
    query = st.text_input("Please enter a query", key="query", on_change=send_input)
    send_button = st.button("Send", key="send_btn")

    # Voice input
    voice_recording = speech_to_text(language="en", use_container_width=True, just_once=True, key="STT")
    if voice_recording:
        query = voice_recording

    # Chat logic
    if query or voice_recording:
        with st.spinner("Processing... Please wait!"):
            response = chain.stream({'question': query})
            print(response)

        # Store messages
        st.session_state.messages.append(("user", query))
        st.session_state.messages.append(("ai", response))

    # Display messages
    with chat_container:
        for role, message in st.session_state.messages:
            st.chat_message(role).write(message)

# Call this in your main script
if __name__ == "__main__":
    run_chatbot_app()
