import asyncio
from pathlib import Path

import openai
import streamlit as st
from streamlit_mic_recorder import speech_to_text

from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM

# ——— Streamlit page config ———
st.set_page_config(page_title="Musk ChatBot | Ask Elon-Level Questions")
st.title("Ask Anything About Musk")

# ——— Secrets & clients ———
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["google_api_key"]

# OpenAI sync client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# TTS settings
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "echo"
SPEECH_FILE = Path(__file__).parent / "speech.mp3"

# ——— Initialize LLM chain ———
llm = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain = create_expert_chain(llm, retriever)

# ——— Session state for chat history ———
if "messages" not in st.session_state:
    st.session_state.messages = []

# ——— UI inputs ———
query = st.text_input("Enter your question or use the mic below:", key="query")
send_button = st.button("Send")
voice_recording = speech_to_text(
    language="en",
    just_once=True,
    use_container_width=True,
    key="STT"
)

# Determine input type
if voice_recording:
    query = voice_recording
    st.markdown(f"🎤 **You said:** {query}")
    is_voice = True
else:
    is_voice = False

# ——— Main handler ———
if (send_button or is_voice) and query:
    with st.spinner("Processing... Please wait!"):
        # Use chain.run to get full text response
        ai_text = chain.run({"question": query})

    # Store user & AI messages for text-chat history
    if not is_voice:
        st.session_state.messages.append(("user", query))
        st.session_state.messages.append(("ai", ai_text))

    # If input was voice, generate and play audio only
    if is_voice:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=ai_text,
            instructions="Speak in a confident, uplifting tone."
        ) as resp:
            resp.stream_to_file(SPEECH_FILE)
        st.audio(str(SPEECH_FILE), format="audio/mp3")

# ——— Render chat history (text only) ———
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)

