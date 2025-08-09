import openai
import streamlit as st
from pathlib import Path
from streamlit_mic_recorder import mic_recorder
import io
import os
import time

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
st.title("🎙️ Musk ChatBot")
st.write("Record your question and hear Elon-level voice responses!")

# Create recording button
st.write("### Record your question:")
audio_bytes = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key=f"recorder_{st.session_state.turn}",
    format="wav"
)

# Process audio if recorded
if audio_bytes:
    with st.spinner("Processing your voice..."):
        # Save user audio to file
        user_audio_path = Path(__file__).parent / f"user_input_{st.session_state.turn}.wav"
        with open(user_audio_path, "wb") as f:
            f.write(audio_bytes['bytes'])
        
        # Transcribe audio using Whisper
        audio_file = io.BytesIO(audio_bytes['bytes'])
        audio_file.name = "recording.wav"
        
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            ).text
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            transcript = ""
        
        # Check for empty/null transcript
        if not transcript or transcript.strip() == "":
            st.warning("🔇 Couldn't detect speech. Please try recording again.")
            # Clean up empty audio file
            if os.path.exists(user_audio_path):
                os.remove(user_audio_path)
            # Skip further processing
            st.stop()
        
        # Add to conversation history
        st.session_state.conversation.append({
            "role": "user",
            "audio": str(user_audio_path),
            "text": transcript
        })
        
        # Generate AI response
        with st.spinner("Generating Your Response..."):
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
if st.session_state.conversation:
    for idx, message in enumerate(st.session_state.conversation):
        if message["role"] == "bot":
            # Left column for bot responses
            with st.container():
                col1, col2 = st.columns([2, 2])
                with col1:
                    st.markdown("**Elon Bot:**")
                    st.audio(message["audio"], format="audio/mp3")
                with col2:
                    with st.expander("See transcript"):
                        st.write(message["text"])
                st.markdown("---")
        else:
            # Right column for user questions
            with st.container():
                col1, col2 = st.columns([3, 3])
                with col2:
                    st.markdown("**You:**")
                    st.audio(message["audio"], format="audio/wav")
                with col1:
                    with st.expander("See transcript"):
                        st.write(message["text"])
                st.markdown("---")
else:
    st.write("No conversation yet. Record your first question above!")

# Clear conversation button
if st.button("Clear Conversation"):
    for msg in st.session_state.conversation:
        if os.path.exists(msg["audio"]):
            os.remove(msg["audio"])
    st.session_state.conversation = []
    st.session_state.turn = 0
    st.rerun()
