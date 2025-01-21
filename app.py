import streamlit as st
from pineconedb import manage_pinecone_store
from chainCreation import create_expert_chain
#call the function to create the chain
from dotenv import load_dotenv
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Load environment variables from .env file
load_dotenv()

retriever=manage_pinecone_store()

chain=create_expert_chain()
# Build the chain
# Set the title of the app
st.title("Ask Anything About Elon Musk")

# Initialize components
history=[]
# Chat container to display conversation
chat_container = st.container()
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_input():
    st.session_state.send_input=True
# Input field for queries

with st.container():
    query = st.text_input("Please enter a query", key="query", on_change=send_input)
    send_button = st.button("Send", key="send_btn")  # Single send button

# Chat logic
if send_button or send_input and query:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response =chain.invoke({'question': query})
        print(response)
    query="user_question:"+query
    response="ai_response:"+response
    history.append((query, response))
    # Generate response
    # Update session state with user query and AI response
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))

with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)
