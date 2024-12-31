from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Union
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize router
router = APIRouter()

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536  # Default dimensions for text-embedding-3-small

class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]
    dimensions: int = DIMENSIONS  # Optional: Allow dimension reduction

def clean_text(text: str) -> str:
    """Clean text by replacing newlines with spaces"""
    return text.replace("\n", " ")

@router.post("/embed")
async def create_embedding(request: EmbeddingRequest):
    """
    Create embeddings for the input text using OpenAI's text-embedding-3-small model.
    
    Args:
        text: Single string or list of strings to embed
        dimensions: Optional number of dimensions (default: 1536)
    
    Returns:
        Embeddings for the input text(s) and usage information
    """
    try:
        # Handle both single text and list of texts
        texts = request.text if isinstance(request.text, list) else [request.text]
        
        # Clean texts
        texts = [clean_text(t) for t in texts]
        
        # Create embeddings
        response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
            dimensions=request.dimensions
        )
        
        # Format response
        result = {
            "object": response.object,
            "data": [{
                "object": item.object,
                "index": item.index,
                "embedding": item.embedding
            } for item in response.data],
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating embedding: {str(e)}")

@router.get("/info")
async def get_model_info():
    """
    Get information about the embedding model being used.
    """
    return JSONResponse(
        content={
            "model": EMBEDDING_MODEL,
            "default_dimensions": DIMENSIONS,
            "max_input_tokens": 8191,
            "description": "OpenAI's text-embedding-3-small model for creating text embeddings",
            "use_cases": [
                "Semantic search",
                "Text clustering",
                "Text classification",
                "Recommendations",
                "Anomaly detection"
            ],
            "performance": {
                "mteb_eval": "62.3%",
                "cost_efficiency": "~62,500 pages per dollar"
            }
        },
        status_code=200
    )
