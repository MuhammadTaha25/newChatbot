import openai
import streamlit as st
from pathlib import Path
from streamlit_mic_recorder import speech_to_text

from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM

# ——— Page config ———
st.set_page_config(page_title="Musk ChatBot (Voice Only)")

# ——— Secrets & clients ———
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY   = st.secrets["google_api_key"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ——— LLM & Pinecone setup ———
llm       = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain     = create_expert_chain(llm, retriever)

# ——— Session‐state for audio history ———
if "bot_audio_files" not in st.session_state:
    st.session_state.bot_audio_files = []
if "user_audio_files" not in st.session_state:
    st.session_state.user_audio_files = []
if "turn" not in st.session_state:
    st.session_state.turn = 0

# ——— UI controls ———
st.title("🎙️ Musk ChatBot (Voice Only)")
st.write("Speak below and hear Elon-level answers back!")

# 1) Get voice → text (and eventually raw audio)
voice_text = speech_to_text(
    language="en",
    just_once=True,
    use_container_width=True,
    key="STT"
)

# TODO: if your recorder can return the raw audio bytes/path, capture it here:
# user_audio = <raw_audio_bytes_or_path>
# st.session_state.user_audio_files.append(user_audio)

if voice_text:
    st.markdown(f"**You said:** _{voice_text}_")  # optional debug
    st.session_state.turn += 1
    turn = st.session_state.turn

    # ——— Generate AI response text ———
    with st.spinner("Thinking…"):
        result = chain.stream({"question": voice_text})
        # if it yields chunks, join them:
        if hasattr(result, "__iter__") and not isinstance(result, str):
            ai_text = "".join(result)
        else:
            ai_text = str(result)

    # ——— Generate TTS and save to unique file ———
    out_path = Path(__file__).parent / f"bot_response_{turn}.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="echo",
        input=ai_text,
        instructions="Speak in a confident, uplifting tone."
    ) as resp:
        resp.stream_to_file(out_path)

    # ——— Push to history ———
    st.session_state.bot_audio_files.append(str(out_path))

# ——— Finally: render the _voice_ history ———
st.markdown("### Conversation so far (voice playback)")

for idx, bot_file in enumerate(st.session_state.bot_audio_files, start=1):
    st.markdown(f"**Turn {idx}:**")
    # (If you captured user audio, you could do st.audio(user_file) here first)
    # st.audio(st.session_state.user_audio_files[idx-1], format="audio/wav")
    st.audio(bot_file, format="audio/mp3")


