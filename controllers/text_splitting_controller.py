from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import re
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

router = APIRouter()

class TextSplitRequest(BaseModel):
    text: str
    chunk_size: Optional[int] = 500  # Default chunk size in characters
    chunk_overlap: Optional[int] = 50  # Default overlap between chunks
    split_method: Optional[str] = "simple"  # Options: "simple", "sentence"

class TextChunk(BaseModel):
    chunk_id: int
    content: str
    metadata: dict

@router.post("/split-text")
async def split_text(request: TextSplitRequest):
    """
    Split text into chunks using specified method and parameters.
    This is a placeholder implementation that will be enhanced with more sophisticated
    chunking mechanisms (e.g., semantic chunking, embedding-based chunking).
    
    Methods available:
    - simple: Split by character count
    - sentence: Split by sentences while respecting chunk size
    
    Future enhancements:
    - Semantic chunking based on content understanding
    - Embedding-based chunking for better context preservation
    - Support for different document types (PDF, HTML, etc.)
    - Integration with vector stores (Pinecone, etc.)
    """
    try:
        chunks = []
        
        if request.split_method == "sentence":
            # Split text into sentences first
            sentences = sent_tokenize(request.text)
            current_chunk = ""
            chunk_id = 0
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > request.chunk_size:
                    if current_chunk:
                        chunks.append(TextChunk(
                            chunk_id=chunk_id,
                            content=current_chunk.strip(),
                            metadata={
                                "method": "sentence",
                                "size": len(current_chunk),
                                "sentence_count": len(sent_tokenize(current_chunk))
                            }
                        ))
                        chunk_id += 1
                        # Keep last part for overlap
                        current_chunk = current_chunk[-request.chunk_overlap:] if request.chunk_overlap > 0 else ""
                
                current_chunk += " " + sentence
            
            # Add the last chunk if there's any content left
            if current_chunk:
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    content=current_chunk.strip(),
                    metadata={
                        "method": "sentence",
                        "size": len(current_chunk),
                        "sentence_count": len(sent_tokenize(current_chunk))
                    }
                ))
        
        else:  # simple method
            text = request.text
            chunk_id = 0
            
            while text:
                # Extract chunk with overlap
                chunk = text[:request.chunk_size]
                if len(text) > request.chunk_size:
                    # Try to break at a space to avoid word splitting
                    last_space = chunk.rfind(" ")
                    if last_space > 0:
                        chunk = chunk[:last_space]
                
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    content=chunk.strip(),
                    metadata={
                        "method": "simple",
                        "size": len(chunk)
                    }
                ))
                
                # Move to next chunk with overlap
                text_pos = len(chunk)
                if request.chunk_overlap > 0:
                    text_pos = max(0, text_pos - request.chunk_overlap)
                text = text[text_pos:].strip()
                chunk_id += 1

        return JSONResponse(
            content={
                "chunks": [chunk.dict() for chunk in chunks],
                "total_chunks": len(chunks),
                "original_length": len(request.text),
                "settings": {
                    "method": request.split_method,
                    "chunk_size": request.chunk_size,
                    "chunk_overlap": request.chunk_overlap
                }
            },
            status_code=200
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error splitting text: {str(e)}")

@router.get("/available-methods")
async def get_available_methods():
    """
    Get information about available text splitting methods and their parameters.
    """
    return JSONResponse(
        content={
            "methods": {
                "simple": {
                    "description": "Split text by character count",
                    "parameters": {
                        "chunk_size": "Number of characters per chunk",
                        "chunk_overlap": "Number of characters to overlap between chunks"
                    }
                },
                "sentence": {
                    "description": "Split text by sentences while respecting chunk size",
                    "parameters": {
                        "chunk_size": "Maximum characters per chunk",
                        "chunk_overlap": "Number of characters to overlap between chunks"
                    }
                }
            },
            "future_methods": [
                "semantic",
                "embedding-based",
                "document-aware",
                "hybrid"
            ]
        },
        status_code=200
    )
