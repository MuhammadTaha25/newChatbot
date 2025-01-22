from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer

# Function to load and parse a specific section of a Wikipedia page
def load_wikipedia_content(url, class_name):
    """
    Load and parse a specific section of a Wikipedia page.

    Parameters:
        url (str): The URL of the Wikipedia page to load.
        class_name (str): The class name of the section to parse.

    Returns:
        list: A list of parsed documents from the specified section of the page.
    """
    # Initialize a WebBaseLoader with the given URL and class name for selective parsing
    loader = WebBaseLoader(
        url,
        bs_kwargs=dict(parse_only=SoupStrainer(class_=(class_name)))  # Use SoupStrainer for targeted parsing
    )
    # Load and return the parsed content as a list of documents
    return loader.load()

