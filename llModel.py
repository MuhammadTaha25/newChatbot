from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

langsmith_tracing='true'
langsmith_endpoint="https://api.smith.langchain.com"
langsmith_api_key="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"
langsmith_project="muskchatbot"
OPENAI_API_KEY=st.secrets
load_dotenv()
def initialize_LLM(openai_api_key=None, gemini_api_key=None):
    """
    Initialize a Language Learning Model (LLM) using OpenAI or Gemini based on the availability of API keys.

    Parameters:
        openai_api_key (str, optional): Your OpenAI API key. Defaults to None and uses the environment variable if not provided.
        gemini_api_key (str, optional): Your Gemini API key. Defaults to None and uses the environment variable if not provided.

    Returns:
        object: An instance of ChatOpenAI (OpenAI model) or GoogleGenerativeAI (Gemini model).
    """
    # Use explicitly provided API keys or fallback to environment variables
    openai_api_key = openai_api_key or OPENAI_API_KEY
    gemini_api_key = gemini_api_key or GOOGLE_API_KEY

# response = openai.ChatCompletion.create(
#     model="gpt-4-turbo",
#     messages=[{"role": "user", "content": "Tell me a joke"}],
    
# )

# for chunk in response:
#     print(chunk["choices"][0]["delta"].get("content", ""), end="")

    if openai_api_key:
        try:
            model_name = "gpt-3.5-turbo"
            LLM = ChatOpenAI(
                model_name=model_name,
                openai_api_key=openai_api_key,
                temperature=0,
                # stream=True
            )
            print("Using OpenAI's GPT-4 model.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI model: {e}")
    elif gemini_api_key:
        try:
            model_name = "gemini-1.5-flash-002"
            LLM = GoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key
            )
            print("Using Gemini's model.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {e}")
    else:
        raise ValueError("No API keys provided. Please set the OpenAI or Gemini API key.")

    return LLM
