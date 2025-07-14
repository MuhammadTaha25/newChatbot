import streamlit as st
from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM
from streamlit_mic_recorder import speech_to_text

# --- Initialization ---
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
GOOGLE_API_KEY = st.secrets['google_api_key']
LLM = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain = create_expert_chain(LLM, retriever)

st.title("Ask anything about Musk")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "send_input" not in st.session_state:
    st.session_state.send_input = False

def mark_send():
    st.session_state.send_input = True

# --- Inputs ---
query = st.text_input("Please enter a query", key="query", on_change=mark_send)
voice = speech_to_text(language="en", just_once=True, use_container_width=True, key="STT")
if voice:
    query = voice
    st.session_state.send_input = True

if st.button("Send", key="send_btn"):
    mark_send()

# --- Processing & UI feedback ---
if st.session_state.send_input and query:
    # 1) Show placeholder text
    status = st.empty()
    status.info("🔍 Retrieving relevant data…")

    # 2) Then spinner + actual chain call
    with st.spinner("Processing... Please wait!"):
        resp_stream = chain.stream({'question': query})
        if hasattr(resp_stream, "__iter__"):
            response = "".join(chunk for chunk in resp_stream)
        else:
            response = resp_stream

    # 3) Clear the “retrieving” message
    status.empty()

    # 4) Append to chat history
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))
    st.session_state.send_input = False

# --- Display chat history ---
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)

