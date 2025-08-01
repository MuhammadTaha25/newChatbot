import openai
import streamlit as st
from pathlib import Path
from streamlit_mic_recorder import mic_recorder, speech_to_text
import time
import os

from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM

# ——— Page config ———
st.set_page_config(page_title="Musk ChatBot (Voice Only)", layout="wide")

# ——— Secrets & clients ———
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["google_api_key"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ——— LLM & Pinecone setup ———
llm = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain = create_expert_chain(llm, retriever)

# ——— Session state for conversation history ———
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "turn" not in st.session_state:
    st.session_state.turn = 0

# ——— UI controls ———
st.title("🎙️ Musk ChatBot (Voice Only)")
st.write("Speak below and hear Elon-level answers back!")

# Create two columns: left for bot, right for user
col_bot, col_user = st.columns([5, 5])

# Voice recording in user column
with col_user:
    st.subheader("Your Voice Input")
    audio_bytes = mic_recorder(
        start_prompt="🎤 Speak",
        stop_prompt="⏹️ Stop",
        key=f"recorder_{st.session_state.turn}"
    )

# Process audio if recorded
if audio_bytes:
    # Save user audio to file
    user_audio_path = Path(__file__).parent / f"user_input_{st.session_state.turn}.wav"
    with open(user_audio_path, "wb") as f:
        f.write(audio_bytes['bytes'])
    
    # Transcribe audio
    with st.spinner("Transcribing..."):
        transcript = speech_to_text(
            audio_bytes=audio_bytes['bytes'],
            language='en',
            key=f"stt_{st.session_state.turn}"
        )
    
    if transcript:
        # Add to conversation history
        st.session_state.conversation.append({
            "role": "user",
            "audio": str(user_audio_path),
            "text": transcript
        })
        
        # Generate AI response
        with st.spinner("Elon is thinking..."):
            result = chain.stream({"question": transcript})
            ai_text = "".join(result) if hasattr(result, "__iter__") else str(result)
        
        # Generate TTS
        bot_audio_path = Path(__file__).parent / f"bot_response_{st.session_state.turn}.mp3"
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="echo",
            input=ai_text,
        ) as resp:
            resp.stream_to_file(bot_audio_path)
        
        # Add bot response to history
        st.session_state.conversation.append({
            "role": "bot",
            "audio": str(bot_audio_path),
            "text": ai_text
        })
        
        st.session_state.turn += 1
        st.rerun()

# ——— Display conversation history ———
st.markdown("## Conversation History")
for idx, message in enumerate(st.session_state.conversation):
    if message["role"] == "bot":
        with st.chat_message("AI", avatar="🚀"):
            st.markdown(f"**Elon Bot:**")
            st.audio(message["audio"], format="audio/mp3")
            with st.expander("See transcript"):
                st.write(message["text"])
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**You:**")
            st.audio(message["audio"], format="audio/wav")
            with st.expander("See transcript"):
                st.write(message["text"])

# Optional: Clean up old audio files
if st.button("Clear Conversation"):
    for msg in st.session_state.conversation:
        if os.path.exists(msg["audio"]):
            os.remove(msg["audio"])
    st.session_state.conversation = []
    st.session_state.turn = 0
    st.rerun()
