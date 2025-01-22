# Elon Musk Q&A Chatbot using Streamlit and Pinecone

## Project Overview
This project implements a conversational AI chatbot that can answer questions about Elon Musk's life, career, companies, and achievements. The system uses a combination of advanced AI models, Pinecone for document retrieval, and Streamlit for the user interface. The chatbot answers questions based on the relevant context extracted from documents related to Elon Musk.

## Features:
- **Voice Input**: Allows users to ask questions via voice using the `speech_to_text` function.
- **Text Input**: Users can type in their queries.
- **Expert Chatbot**: The chatbot responds as an expert on Elon Musk, leveraging relevant context from the Pinecone vector store.
- **Contextual Responses**: Uses advanced embeddings and LLMs (OpenAI and Gemini) to generate answers.

## Project Components

### 1. `initialize_LLM()` (from `llModel.py`)
This function initializes the language learning model (LLM) for generating responses to user queries. It supports both OpenAI and Gemini models.

**Parameters:**
- `openai_api_key`: Your OpenAI API key (optional, defaults to environment variable).
- `gemini_api_key`: Your Gemini API key (optional, defaults to environment variable).

**Returns:**
- An instance of `ChatOpenAI` (OpenAI) or `GoogleGenerativeAI` (Gemini).

### 2. `manage_pinecone_store()` (from `pineconedb.py`)
This function manages the Pinecone vector store, either by loading an existing index or creating a new one with the processed documents.

**Parameters:**
- `index_name`: Pinecone index name (from environment secrets).
- `embeddings`: The embedding model used for generating vector representations.

**Returns:**
- A retriever object for fetching relevant document chunks from the Pinecone index.

### 3. `create_expert_chain()` (from `creating_chain.py`)
This function creates a question-answer chain using the initialized LLM and the retriever (from Pinecone). The chain is configured to answer questions specifically about Elon Musk.

**Parameters:**
- `LLM`: The initialized language model (either OpenAI or Gemini).
- `retriever`: The retriever object for fetching document chunks.

**Returns:**
- A chain object that uses the LLM and retriever to generate expert answers.

### 4. `speech_to_text()` (from `streamlit_mic_recorder`)
This function converts voice input into text, enabling voice queries from users.

**Parameters:**
- `language`: Language of the voice input (default: English).
- `use_container_width`: Use container width for layout (optional).
- `just_once`: Capture the voice input only once (optional).

**Returns:**
- The converted text from the recorded speech.

### 5. Streamlit
Streamlit is used to create the web interface for the chatbot.

**Key Components:**
- `st.title()`: Sets the title of the Streamlit app.
- `st.text_input()`: Accepts text input from the user for queries.
- `st.button()`: A button for submitting queries.
- `st.session_state`: Stores chat messages and manages the state of the conversation.
- `st.spinner()`: Displays a loading spinner while processing user input.
- `st.chat_message()`: Displays messages in a conversational chat format.

## Usage
- The app will prompt you to enter a query either through text or voice input.
- You can ask questions about Elon Musk, and the chatbot will provide responses based on the retrieved context.
- If the query is outside the context of Elon Musk, the chatbot will reply: "I am trained to answer questions related to Elon Musk only."
