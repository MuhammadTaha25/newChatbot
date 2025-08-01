# app.py
import openai
import streamlit as st
from pathlib import Path

# Install: pip install streamlit-audiorecorder
from streamlit_audiorecorder import audiorecorder

from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM

# ——— Page config ———
st.set_page_config(page_title="Musk ChatBot (Voice-Only with History)")

# ——— Secrets & clients ———
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY   = st.secrets["google_api_key"]
openai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)

# ——— LLM & Pinecone setup ———
llm       = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain     = create_expert_chain(llm, retriever)

# ——— Session-state for history ———
if "history" not in st.session_state:
    # each entry: dict with keys user_audio, bot_audio
    st.session_state.history = []
if "turn" not in st.session_state:
    st.session_state.turn = 0

st.title("🎙️ Musk ChatBot")
st.write("Record your question below, then listen to Elon-level answers!")

# ——— 1) Record user audio ———
user_bytes = audiorecorder("Click to record", "Recording...")

if user_bytes:
    st.session_state.turn += 1
    turn = st.session_state.turn

    # 2) Save raw user audio to disk
    user_path = Path(f"turn_{turn}_user.wav")
    with open(user_path, "wb") as f:
        f.write(user_bytes)

    # 3) Transcribe via OpenAI Whisper
    with open(user_path, "rb") as audio_file:
        transcript_resp = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    user_text = transcript_resp["text"]
    st.markdown(f"**You said:** *{user_text}*")

    # 4) Ask your chain
    with st.spinner("Thinking…"):
        # handle stream or static return
        result = chain.stream({"question": user_text})
        if hasattr(result, "__iter__") and not isinstance(result, str):
            ai_text = "".join(result)
        else:
            ai_text = str(result)

    # 5) Generate TTS for bot
    bot_path = Path(f"turn_{turn}_bot.mp3")
    with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="echo",
        input=ai_text,
        # instructions="Speak in a confident, uplifting tone."
    ) as resp:
        resp.stream_to_file(bot_path)

    # 6) Append to history
    st.session_state.history.append({
        "user_audio": str(user_path),
        "bot_audio": str(bot_path)
    })

# ——— Render history side-by-side ———
st.markdown("## Conversation History")
for idx, turn_data in enumerate(st.session_state.history, start=1):
    st.markdown(f"**Turn {idx}**")
    left, right = st.columns(2)
    with left:
        st.markdown("**Elon says:**")
        st.audio(turn_data["bot_audio"], format="audio/mp3")
    with right:
        st.markdown("**You said:**")
        st.audio(turn_data["user_audio"], format="audio/wav")
