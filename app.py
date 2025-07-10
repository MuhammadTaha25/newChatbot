from pineconedb import manage_pinecone_store
from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM
import streamlit as st
from streamlit_mic_recorder import speech_to_text
OPENAI_API_KEY =st.secrets['OPENAI_API_KEY']
GOOGLE_API_KEY =st.secrets['google_api_key']

LLM=initialize_LLM(OPENAI_API_KEY,GOOGLE_API_KEY)
retriever=manage_pinecone_store()
chain=create_expert_chain(LLM,retriever)
# Build the chain
# Set the title of the app
# Initialize components
# Chat container to display conversation
st.title("Ask anything about Musk")
chat_container = st.container()
if "messages" not in st.session_state:
    st.session_state.messages = []
def send_input():
    st.session_state.send_input=True
   
query= st.text_input("Please enter a query", key="query", on_change=send_input)
send_button = st.button("Send", key="send_btn")  # Single send button
    
voice_recording=speech_to_text(language="en",use_container_width=True,just_once=True,key="STT")
    
if voice_recording:
    query=voice_recording
    
# Chat logic
if query or voice_recording:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response =chain.stream({'question': query})
        print(response)

    # Generate response
    # Update session state with user query and AI response
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))
    
with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message) 
