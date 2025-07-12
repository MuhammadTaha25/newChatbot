from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM
import streamlit as st
from streamlit_mic_recorder import speech_to_text

# — LangSmith config (if you actually use it) —
langsmith_tracing  = "true"
langsmith_endpoint = "https://api.smith.langchain.com"
langsmith_api_key  = "your_langsmith_api_key"
langsmith_project  = "muskchatbot"

# Load your keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY  = st.secrets["google_api_key"]

# Initialize LLM, retriever, chain
LLM       = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain     = create_expert_chain(LLM, retriever)

# Streamlit UI
st.title("Ask anything about Musk")
chat_container = st.container()
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_input():
    st.session_state.send_input = True

# Text input + send button
query = st.text_input("Please enter a query", key="query", on_change=send_input)
st.button("Send", key="send_btn")

# Optional voice input
voice_recording = speech_to_text(
    language="en",
    use_container_width=True,
    just_once=True,
    key="STT"
)
if voice_recording:
    query = voice_recording

# Only run the chain when user actually submits something
if (query and st.session_state.send_input) or voice_recording:
    with st.spinner("Processing... Please wait!"):
        # 1) kick off the stream
        response_stream = chain.stream({"question": query})
        # 2) accumulate into a single string
        response_text = ""
        for chunk in response_stream:
            # if chunk is a dict, use .get("answer"); otherwise just add chunk
            if isinstance(chunk, dict):
                response_text += chunk.get("answer", "")
            else:
                response_text += str(chunk)

    # 3) update session state
    with st.container():
        st.session_state.messages.append(("user", query))
        st.session_state.messages.append(("assistant", response_text))
    # reset send flag so it doesn’t re-run
    st.session_state.send_input = False

# Finally, render the chat log
with st.container():
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)
