from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import StrOutputParser
from operator import itemgetter
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
openai_client = wrap_openai(OpenAI())

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

langsmith_tracing='true'
langsmith_endpoint="https://api.smith.langchain.com"
langsmith_api_key="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"
langsmith_project="muskchatbot"

OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']
@traceable
def create_expert_chain(LLM=None, retriever=None):
    """
    Create a chain for answering questions as an expert on Elon Musk.

    Parameters:
        llm (object): The language model to use for generating responses.
        retriever (object): A retriever for fetching relevant context based on the question.

    Returns:
        object: A configured chain for answering questions about Elon Musk.
    """
    # Define the prompt template
    prompt_str = """
You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about Elon Musk.
Answer all questions as if you are an expert on his life, career, companies, and achievements. You are trained to answer question related to the provied
context if a user ask question which is different from the context you have to say  :" I am train to answer questions related to Elon Musk only."
Context: {context}
Question: {question}
    """
    _prompt = ChatPromptTemplate.from_template(prompt_str)

    # Chain setup
    query_fetcher = itemgetter("question")  # Extract the question from input
    setup = {
        "question": query_fetcher,          # Fetch the question from input
        "context": query_fetcher | retriever|format_docs  # Combine the question with the retriever
    }
    _chain = setup | _prompt | LLM | StrOutputParser()

    return _chain
