import os
from pathlib import Path

import openai
import streamlit as st
from streamlit_audiorecorder import audiorecorder  # pip install streamlit-audiorecorder

from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM

# ——— Config & secrets ———
st.set_page_config(page_title="Musk ChatBot (Voice Chat UI)")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY   = st.secrets["google_api_key"]
openai.api_key   = OPENAI_API_KEY

# ——— Setup LLM chain ———
llm = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain = create_expert_chain(llm, retriever)

# ——— Session‐state for storing audio files ———
if "user_audio" not in st.session_state:
    st.session_state.user_audio = []
if "bot_audio" not in st.session_state:
    st.session_state.bot_audio = []
if "turn" not in st.session_state:
    st.session_state.turn = 0

st.title("🎙️ Musk ChatBot (Voice Chat)")

# ——— 1) Record user voice ———
audio_bytes = audiorecorder("Speak now", "Recording...")  
if audio_bytes:
    turn = st.session_state.turn + 1
    tmp_dir = Path(__file__).parent / "audio_history"
    tmp_dir.mkdir(exist_ok=True)
    
    # save user audio
    user_path = tmp_dir / f"user_{turn}.wav"
    with open(user_path, "wb") as f:
        f.write(audio_bytes)
    st.session_state.user_audio.append(str(user_path))

    # transcribe with Whisper
    transcript = openai.Audio.transcribe("whisper-1", user_path)
    question = transcript["text"]
    st.markdown(f"**You:** _{question}_")
    
    # call chain
    with st.spinner("Thinking…"):
        # collect streamed chunks if needed
        out = chain.stream({"question": question})
        if hasattr(out, "__iter__") and not isinstance(out, str):
            answer = "".join(out)
        else:
            answer = str(out)
    
    # generate TTS
    bot_path = tmp_dir / f"bot_{turn}.mp3"
    with openai.OpenAI().audio.speech.with_streaming_response.create(
        model="tts-1-hd",
        voice="echo",
        input=answer,
        # instructions="Speak in a confident, uplifting tone."
    ) as resp:
        resp.stream_to_file(bot_path)
    st.session_state.bot_audio.append(str(bot_path))
    
    # increment turn
    st.session_state.turn = turn

# ——— 2) Render chat history as alternating audio players ———
for i in range(st.session_state.turn):
    st.markdown(f"**Turn {i+1}:**")
    st.audio(st.session_state.user_audio[i], format="audio/wav")
    st.audio(st.session_state.bot_audio[i],  format="audio/mp3")
