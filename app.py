from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from pineconedb import manage_pinecone_store
from llModel import initialize_LLM
import streamlit as st
from utils import get_answer, text_to_speech, autoplay_audio, speech_to_text
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

float_init()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How may I assist you today?"}
        ]

initialize_session_state()

st.title("Ask Anything About Elon Musk")

footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()
# Load environment variables from .env file

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    # Write the audio bytes to a file
    with st.spinner("Transcribing..."):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages)
        with st.spinner("Generating audio response..."):    
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

# Float the footer container and provide CSS to target it with
footer_container.float("bottom: 0rem;")
OPENAI_API_KEY =st.secrets['OPENAI_API_KEY']
GOOGLE_API_KEY =st.secrets['google_api_key']

LLM=initialize_LLM(OPENAI_API_KEY,GOOGLE_API_KEY)
retriever=manage_pinecone_store()
chain=create_expert_chain(LLM,retriever)
# Build the chain
# Set the title of the app
st.title("Ask Anything About Elon Musk")

# Initialize components
# Chat container to display conversation
chat_container = st.container()
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_input():
    st.session_state.send_input=True
# Input field for queries

with st.container():
    query = st.text_input("Please enter a query", key="query", on_change=send_input)
    send_button = st.button("Send", key="send_btn")  # Single send button

# Chat logic
if send_button or send_input and query:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response =chain.invoke({'question': query})
        print(response)

    # Generate response
    # Update session state with user query and AI response
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))

with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)
