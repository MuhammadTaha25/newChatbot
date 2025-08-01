from pathlib import Path

import openai
import streamlit as st
from streamlit_mic_recorder import speech_to_text

from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM

# ——— Page config ———
st.set_page_config(page_title="Musk ChatBot | Ask Elon-Level Questions")
st.title("Ask Anything About Musk")

# ——— Secrets & clients ———
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["google_api_key"]

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# TTS settings
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "echo"
SPEECH_FILE = Path(__file__).parent / "speech.mp3"

# ——— Build the chain ———
llm = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain = create_expert_chain(llm, retriever)

# ——— Chat history ———
if "messages" not in st.session_state:
    st.session_state.messages = []

# ——— Inputs ———
query = st.text_input("Type your question or use the mic below:", key="query")
send_button = st.button("Send")

voice_recording = speech_to_text(
    language="en",
    just_once=True,
    use_container_width=True,
    key="STT"
)
if voice_recording:
    query = voice_recording
    st.markdown(f"🎤 **You said:** {query}")
    is_voice = True
else:
    is_voice = False

# ——— Main logic ———
if (send_button or is_voice) and query:
    with st.spinner("Thinking..."):
        # invoke the RunnableSequence synchronously :contentReference[oaicite:0]{index=0}
        result = chain.invoke({"question": query})
        # unpack if it returned a dict
        if isinstance(result, dict):
            ai_text = result.get("text") or result.get("answer") or list(result.values())[-1]
        else:
            ai_text = str(result)

    # for text inputs: show chat bubbles
    if not is_voice:
        st.session_state.messages.append(("user", query))
        st.session_state.messages.append(("ai", ai_text))

    # for voice inputs: generate & play TTS only
    if is_voice:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=ai_text,
            instructions="Speak in a confident, uplifting tone."
        ) as resp:
            resp.stream_to_file(SPEECH_FILE)
        st.audio(str(SPEECH_FILE), format="audio/mp3")

# ——— Render text chat history ———
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)


