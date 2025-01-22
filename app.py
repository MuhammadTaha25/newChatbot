from pineconedb import manage_pinecone_store  # Import Pinecone store management function
from creating_chain import create_expert_chain  # Import function to create expert Q&A chain
from llModel import initialize_LLM  # Import function to initialize the LLM (OpenAI or Gemini)
import streamlit as st  # Import Streamlit for building the web app interface
from streamlit_mic_recorder import speech_to_text  # Import function for speech-to-text conversion

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']  # Fetch OpenAI API key from secrets
GOOGLE_API_KEY = st.secrets['google_api_key']  # Fetch Google API key from secrets

# Initialize LLM (language model) and retriever for question answering
LLM = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)  # Initialize the LLM model
retriever = manage_pinecone_store()  # Retrieve Pinecone vector store for relevant document chunks
chain = create_expert_chain(LLM, retriever)  # Create a chain that uses the LLM and retriever for expert Q&A

# Set up Streamlit interface components
st.title("Ask anything about Musk 🤖")  # Display the app title
chat_container = st.container()  # Define a container to hold chat messages

if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize messages in session state if not present

def send_input():
    st.session_state.send_input = True  # Set flag when input is sent

# Text input field for user query
query = st.text_input("Please enter a query", key="query", on_change=send_input)
send_button = st.button("Send", key="send_btn")  # Send button for submitting query

# Voice input recording using speech-to-text
with st.container():
    voice_recording = speech_to_text(language="en", use_container_width=True, just_once=True, key="STT")

if voice_recording:
    query = voice_recording  # Use voice recording as query input

# Handle query and generate response
if query or voice_recording:
    with st.spinner("Processing... Please wait!"):  # Show spinner while processing
        response = chain.invoke({'question': query})  # Invoke the chain to generate the response
        print(response)

    # Update session state with user query and AI response
    st.session_state.messages.append(("user", query))  # Add user message
    st.session_state.messages.append(("ai", response))  # Add AI response

# Display chat messages in the container
with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)  # Display messages as chat bubbles
