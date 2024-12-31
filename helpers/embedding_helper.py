from openai import OpenAI
from typing import List, Union
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536

def clean_text(text: str) -> str:
    """Clean text by replacing newlines with spaces and removing extra whitespace"""
    return ' '.join(text.replace('\n', ' ').split())

def create_embedding(text: Union[str, List[str]], dimensions: int = DEFAULT_DIMENSIONS) -> Union[List[float], List[List[float]]]:
    """
    Create embeddings for text using OpenAI's text-embedding-3-small model.
    
    Args:
        text: Single string or list of strings to embed
        dimensions: Optional number of dimensions (default: 1536)
    
    Returns:
        List of embeddings if input is a list, single embedding if input is a string
    """
    # Clean and prepare text
    if isinstance(text, str):
        texts = [clean_text(text)]
    else:
        texts = [clean_text(t) for t in text]
    
    # Create embeddings
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
        dimensions=dimensions
    )
    
    # Extract embeddings
    embeddings = [item.embedding for item in response.data]
    
    # Return single embedding for single text, list of embeddings for multiple texts
    return embeddings[0] if isinstance(text, str) else embeddings
