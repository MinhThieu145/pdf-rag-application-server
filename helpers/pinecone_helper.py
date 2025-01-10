from pinecone import Pinecone
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import itertools
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = None
index = None

def init_pinecone():
    global pc, index
    if pc is not None:
        return

    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    # Get the index name from the host
    index_name = os.getenv('PINECONE_HOST')
    logger.info(f"Connecting to Pinecone index: {index_name}")

    try:
        # First, check if index exists
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI's text-embedding-3-small dimension
                metric="cosine"
            )
        
        # Get the index
        index = pc.Index(index_name)
        logger.info("Successfully connected to Pinecone index")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone index: {str(e)}")
        raise

def chunks(iterable: List[Any], batch_size: int = 100):
    """Break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def prepare_vectors_for_upsert(content_items: List[Dict[str, Any]], pdf_name: str) -> List[Dict[str, Any]]:
    """
    Prepare vectors for upserting to Pinecone.
    Each text chunk becomes a vector with metadata about its source.
    """
    vectors = []
    logger.info(f"Processing {len(content_items)} items from PDF: {pdf_name}")
    
    for item in content_items:
        # Log item structure to debug
        logger.info(f"Processing item of type: {item.get('type')}")
        logger.info(f"Item keys: {list(item.keys())}")
        
        if item.get('type') in ['text', 'heading']:
            if 'vector_chunks' in item:
                for chunk in item['vector_chunks']:
                    vector_id = f"{pdf_name}_{uuid.uuid4()}"
                    vectors.append({
                        "id": vector_id,
                        "values": chunk['vector_value'],
                        "metadata": {
                            "pdf_name": pdf_name,
                            "text": chunk['text'],
                            "start_char": chunk.get('start_char', 0),
                            "end_char": chunk.get('end_char', 0),
                            "type": item['type']
                        }
                    })
            else:
                logger.warning(f"No vector_chunks found in item of type {item.get('type')}")
                
    logger.info(f"Created {len(vectors)} vectors for PDF: {pdf_name}")
    if vectors:
        # Log first vector as example
        logger.info(f"Example vector: {vectors[0]}")
    return vectors

async def upsert_pdf_vectors(content: List[Dict[str, Any]], pdf_name: str, batch_size: int = 100):
    """
    Upsert vectors from PDF content to Pinecone.
    Uses batching for better performance.
    """
    init_pinecone()
    try:
        logger.info(f"Starting vector upsert for PDF: {pdf_name}")
        
        # Prepare vectors from content
        vectors = prepare_vectors_for_upsert(content, pdf_name)
        if not vectors:
            logger.warning(f"No vectors to upsert for PDF: {pdf_name}")
            return {
                "status": "warning",
                "message": "No vectors found to upsert",
                "vector_count": 0
            }
        
        # Upsert in batches
        total_upserted = 0
        for i, vector_chunk in enumerate(chunks(vectors, batch_size=batch_size)):
            logger.info(f"Upserting batch {i+1} ({len(vector_chunk)} vectors)")
            try:
                index.upsert(vectors=list(vector_chunk))
                total_upserted += len(vector_chunk)
                logger.info(f"Successfully upserted batch {i+1}")
            except Exception as e:
                logger.error(f"Failed to upsert batch {i+1}: {str(e)}")
                raise
        
        logger.info(f"Successfully upserted all {total_upserted} vectors for PDF: {pdf_name}")
        return {
            "status": "success",
            "message": f"Successfully upserted {total_upserted} vectors for PDF: {pdf_name}",
            "vector_count": total_upserted
        }
        
    except Exception as e:
        error_msg = f"Failed to upsert vectors to Pinecone: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

async def search_vectors(query_vector: List[float], filter: Dict[str, str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar vectors in Pinecone.
    
    Args:
        query_vector: The query vector to search with
        filter: Optional metadata filter (e.g., {"pdf_name": "example.pdf"})
        top_k: Number of results to return
        
    Returns:
        List of matches with their metadata
    """
    init_pinecone()
    try:
        # Perform the search
        results = index.query(
            vector=query_vector,
            filter=filter,
            top_k=top_k,
            include_metadata=True
        )
        
        # Return the matches with their metadata
        return results.matches
        
    except Exception as e:
        logger.error(f"Error searching vectors: {str(e)}")
        raise
