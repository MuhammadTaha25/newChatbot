import openai
import streamlit as st
from pathlib import Path
from streamlit_mic_recorder import speech_to_text

from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM

st.set_page_config(page_title="Musk ChatBot (Voice Only)")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY   = st.secrets["google_api_key"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

llm       = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain     = create_expert_chain(llm, retriever)

# ─── session state ───
if "bot_audio_files" not in st.session_state:
    st.session_state.bot_audio_files = []
if "user_audio_files" not in st.session_state:
    st.session_state.user_audio_files = []
if "turn" not in st.session_state:
    st.session_state.turn = 0

st.title("🎙️ Musk ChatBot (Voice Only)")
st.write("Speak below and hear Elon-level answers back!")

# ─── 1) Get voice → text + raw audio path ───
# assume speech_to_text can also return `audio_file` path via return_audio=True
voice_text, user_audio = speech_to_text(
    language="en",
    just_once=True,
    use_container_width=True,
    return_audio=True,      # ← your recorder lib must support this
    key="STT"
)

if voice_text:
    st.markdown(f"**You said:** _{voice_text}_")
    st.session_state.turn += 1
    turn = st.session_state.turn

    # save user audio path
    st.session_state.user_audio_files.append(user_audio)

    # ─── Generate AI response text ───
    with st.spinner("Thinking…"):
        result = chain.stream({"question": voice_text})
        if hasattr(result, "__iter__") and not isinstance(result, str):
            ai_text = "".join(result)
        else:
            ai_text = str(result)

    # ─── Generate TTS ───
    out_path = Path(__file__).parent / f"bot_response_{turn}.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="echo",
        input=ai_text,
        instructions="Speak in a confident, uplifting tone."
    ) as resp:
        resp.stream_to_file(out_path)
    st.session_state.bot_audio_files.append(str(out_path))

# ─── render side-by-side history ───
st.markdown("### Conversation so far (voice playback)")
for idx, (u_file, b_file) in enumerate(zip(st.session_state.user_audio_files,
                                          st.session_state.bot_audio_files), start=1):
    st.markdown(f"**Turn {idx}:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**You:**")
        st.audio(u_file, format="audio/wav")
    with col2:
        st.write("**Bot:**")
        st.audio(b_file, format="audio/mp3")

