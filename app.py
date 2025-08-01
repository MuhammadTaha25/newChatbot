import asyncio
from pathlib import Path
import openai
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM
from openai.helpers import LocalAudioPlayer

#––– Setup keys & clients
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["google_api_key"]

client = openai.OpenAI(api_key=OPENAI_API_KEY)
tts_model = "gpt-4o-mini-tts"
tts_voice = "echo"

LLM = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain = create_expert_chain(LLM, retriever)

#––– Streamlit UI
st.set_page_config(page_title="Musk ChatBot")
st.title("Ask Anything About Musk")

if "messages" not in st.session_state:
    st.session_state.messages = []

def send_input():
    st.session_state.send_input = True

query = st.text_input("Enter query or use mic below", key="query", on_change=send_input)
st.button("Send", on_click=send_input)

voice_recording = speech_to_text(language="en", just_once=True, use_container_width=True, key="STT")
if voice_recording:
    query = voice_recording
    st.write("🎤 You said:", query)

#––– Main logic
if (query and st.session_state.get("send_input")):
    with st.spinner("Processing..."):
        ai_text = chain.stream({"question": query})
    # Reset flag
    st.session_state.send_input = False

    # Store text response
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", ai_text))

    # Generate TTS and play only if input was audio
    if voice_recording:
        # prepare file path
        speech_file = Path(__file__).parent / "speech.mp3"
        # blocking TTS call
        with client.audio.speech.with_streaming_response.create(
            model=tts_model,
            voice=tts_voice,
            input=ai_text,
            instructions="Speak in a confident, uplifting tone."
        ) as resp:
            resp.stream_to_file(speech_file)
        st.audio(str(speech_file))   # Streamlit audio player
    else:
        # text-input case: show text as before
        pass

#––– Render chat history (only text bubbles)
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)
