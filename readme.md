[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.25-orange.svg)](https://streamlit.io)

# MuskChatBot 🤖📄

**MuskChatBot** is a focused, document-driven chatbot that answers questions specifically about Elon Musk by ingesting Wikipedia content (or any uploaded PDFs). Built with **LangChain**, **OpenAI LLMs**, and **vector databases**, it delivers fast, context-aware responses via an intuitive **Streamlit** interface.

---

## 📋 Table of Contents

- [✨ Features](#-features)  
- [🛠 Tech Stack](#-tech-stack)  
- [⚙️ How It Works](#️-how-it-works)  
- [🚀 Quick Start](#-quick-start)  
- [📁 Project Structure](#-project-structure)  
- [🤝 Contributing](#-contributing)  
- [📄 License](#-license)

---

## ✨ Features

- **📰 Wikipedia Integration**: Automatically load and parse Elon Musk’s Wikipedia page  
- **📄 PDF Support**: Upload custom PDFs to extend the knowledge base  
- **❓ Contextual Q&A**: Ask any natural-language question about Musk and get precise answers  
- **⚡ Real-Time Responses**: Leverages OpenAI LLMs for immediate, coherent replies  
- **🔍 Semantic Search**: Retrieval-Augmented Generation (RAG) with vector embeddings  
- **🗃️ Flexible Vector Stores**: Supports FAISS, Chroma, etc.  
- **🧩 Modular Architecture**: Swap loaders, splitters, and stores with minimal changes  

---

## 🛠 Tech Stack

| Component           | Technology                                 |
|---------------------|--------------------------------------------|
| **UI**              | Streamlit                                  |
| **LLM Integration** | OpenAI (via LangChain)                     |
| **Embeddings**      | OpenAIEmbeddings                           |
| **Vector Database** | FAISS / Chroma                             |
| **Data Loaders**    | WikipediaLoader / PyMuPDF / PDFMiner       |
| **Language**        | Python ≥ 3.8                               |

---

## ⚙️ How It Works

1. **Data Ingestion**  
   - Fetch Elon Musk’s Wikipedia content via a web-based loader  
   - (Optional) Upload additional PDFs  

2. **Chunk & Embed**  
   - Split text into manageable chunks (e.g., 500 tokens)  
   - Convert each chunk into vector embeddings  

3. **Indexing**  
   - Store embeddings in the configured vector database  

4. **Query Processing**  
   - Embed the user’s question and perform a semantic similarity search  

5. **Answer Generation**  
   - Retrieve top-k relevant chunks  
   - Pass them to the LLM to craft a concise, context-aware answer  

---

## 🚀 Quick Start

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/MuskChatBot.git
   cd MuskChatBot
