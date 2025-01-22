from langchain.prompts.chat import ChatPromptTemplate  # Importing ChatPromptTemplate to create custom prompts
from langchain.schema import StrOutputParser  # Importing StrOutputParser for formatting the output
from operator import itemgetter  # Importing itemgetter to fetch data from input

def format_docs(docs):
    """
    Format the document content by joining all pages' content.

    Parameters:
        docs (list): List of documents with page content.

    Returns:
        str: Combined string of document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)  # Join page content with double newlines

def create_expert_chain(LLM=None, retriever=None):
    """
    Create a chain for answering questions as an expert on Elon Musk.

    Parameters:
        LLM (object): The language model to use for generating responses.
        retriever (object): A retriever for fetching relevant context based on the question.

    Returns:
        object: A configured chain for answering questions about Elon Musk.
    """
    # Define the prompt template for generating expert answers about Elon Musk
    prompt_str = """
You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about Elon Musk.
Answer all questions as if you are an expert on his life, career, companies, and achievements. You are trained to answer questions related to the provided
context. If a user asks a question unrelated to the context, you will say: "I am trained to answer questions related to Elon Musk only."
Context: {context}
Question: {question}
    """
    _prompt = ChatPromptTemplate.from_template(prompt_str)  # Create the prompt from the template

    # Chain setup to fetch the question and context
    query_fetcher = itemgetter("question")  # Extract the question from the input
    setup = {
        "question": query_fetcher,          # Fetch the question
        "context": query_fetcher | retriever | format_docs  # Combine question with retriever and document content
    }
    _chain = setup | _prompt | LLM | StrOutputParser()  # Set up the chain for response generat
