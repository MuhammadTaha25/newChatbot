import openai
import streamlit as st
from pathlib import Path
from creating_chain import create_expert_chain
from llModel import initialize_LLM
from pineconedb import manage_pinecone_store
from streamlit_audiorecorder import audiorecorder

import tempfile

# Page config
st.set_page_config(page_title="Musk ChatBot (Voice Only)")

# Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["google_api_key"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Init LLM & chain
llm = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain = create_expert_chain(llm, retriever)

# Session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Title
st.title("🎤 Musk ChatBot with Voice Chat")

# Record user voice
st.markdown("#### 🎙️ Record Your Question")
audio_bytes = audiorecorder('Record your query', "click to Record')

if audio_bytes:
    # Save user audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        user_audio_path = f.name

    st.audio(user_audio_path, format="audio/wav")
    
    # Transcribe user voice to text
    with st.spinner("Transcribing your voice..."):
        transcription = client.audio.transcriptions.create(
            file=open(user_audio_path, "rb"),
            model="whisper-1"
        )
        user_text = transcription.text

    st.markdown(f"**You said:** _{user_text}_")

    # Generate response
    with st.spinner("Thinking..."):
        result = chain.stream({"question": user_text})
        ai_text = "".join(result) if hasattr(result, "__iter__") else str(result)

    # TTS for chatbot response
    turn = len(st.session_state.conversation) + 1
    bot_audio_path = Path(f"bot_response_{turn}.mp3")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="echo",
        input=ai_text,
        instructions="Respond confidently and clearly."
    ) as resp:
        resp.stream_to_file(bot_audio_path)

    # Store in session history
    st.session_state.conversation.append({
        "user_audio": user_audio_path,
        "bot_audio": str(bot_audio_path)
    })

# Show conversation history
st.markdown("### 🗣️ Conversation History")

for idx, turn in enumerate(st.session_state.conversation, 1):
    col1, col2 = st.columns(2)
    with col2:
        st.markdown(f"👤 **You (Turn {idx})**")
        st.audio(turn["user_audio"], format="audio/wav")
    with col1:
        st.markdown(f"🤖 **MuskBot**")
        st.audio(turn["bot_audio"], format="audio/mp3")

