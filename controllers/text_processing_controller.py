from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import json
import boto3
from botocore.exceptions import ClientError
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from helpers.embedding_helper import create_embedding, clean_text
from helpers.pinecone_helper import upsert_pdf_vectors
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AWS S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Initialize router
router = APIRouter()

# Constants
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Characters of overlap between chunks

def split_text(text: str) -> List[Dict[str, Any]]:
    """Split text into chunks with overlap"""
    text = clean_text(text)
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk of text
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        
        # If this isn't the last chunk, try to break at a space
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space > 0:
                end = start + last_space
                chunk = text[start:end]
        
        # Create embedding for chunk
        embedding = create_embedding(chunk)
        
        chunks.append({
            "text": chunk,
            "start_char": start,
            "end_char": end,
            "vector_value": embedding
        })
        
        # Move start position for next chunk, accounting for overlap
        start = end - CHUNK_OVERLAP if end < len(text) else len(text)
    
    return chunks

@router.post("/process-pdf-content/{filename}")
async def process_pdf_content(filename: str):
    """
    Process PDF content by:
    1. Reading the JSON from S3
    2. Splitting text into chunks
    3. Creating embeddings for each chunk
    4. Saving the enhanced JSON back to S3
    5. Upserting vectors to Pinecone
    """
    try:
        # Get file info
        clean_filename = filename.split('/')[-1]
        base_name = Path(clean_filename).stem
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        
        # Source and destination paths
        source_key = f"documents/{base_name}/json/{base_name}.json"
        logger.info(f"Reading source JSON from: {source_key}")
        
        # Read original JSON from S3
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=source_key)
            content = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Successfully read JSON content")
            logger.info(f"JSON structure: {list(content[0].keys())}")  
        except ClientError as e:
            error_msg = f"Source content not found: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=404,
                detail=error_msg
            )
        
        # Process each page
        total_items_processed = 0
        pages = content[0].get('pages', [])
        logger.info(f"Found {len(pages)} pages to process")
        
        for page in pages:
            # Process each item in the page
            items_in_page = 0
            items = page.get('items', [])
            logger.info(f"Found {len(items)} items in page {page.get('page', 'unknown')}")
            
            for item in items:
                if item.get('type') in ['text', 'heading'] and item.get('value'):
                    # Split and embed the text
                    text_chunks = split_text(item['value'])
                    logger.info(f"Created {len(text_chunks)} chunks for item of type {item.get('type')}")
                    # Add vector values to the item
                    item['vector_chunks'] = text_chunks
                    items_in_page += 1
            total_items_processed += items_in_page
            logger.info(f"Processed {items_in_page} items in page {page.get('page', 'unknown')}")
        
        logger.info(f"Total items processed: {total_items_processed}")
        
        # Save enhanced JSON to new location
        dest_folder = f"documents/{base_name}/text-split-llama-parse-default"
        dest_key = f"{dest_folder}/{base_name}.json"
        logger.info(f"Saving enhanced JSON to: {dest_key}")
        
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=dest_key,
                Body=json.dumps(content, indent=2)
            )
            logger.info("Successfully saved enhanced JSON")
        except ClientError as e:
            error_msg = f"Failed to save enhanced content: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
            
        # Upsert vectors to Pinecone
        try:
            # Flatten the items from all pages into a single list
            all_items = []
            for page in content[0]['pages']:
                all_items.extend(page.get('items', []))
            logger.info(f"Sending {len(all_items)} items to Pinecone")
            
            pinecone_result = await upsert_pdf_vectors(
                content=all_items,  # Pass flattened items list instead of pages
                pdf_name=base_name
            )
        except Exception as e:
            error_msg = f"Failed to upsert vectors to Pinecone: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
        
        return JSONResponse(
            content={
                "message": "PDF content processed successfully",
                "source": source_key,
                "destination": dest_key,
                "filename": filename,
                "items_processed": total_items_processed,
                "pinecone_status": pinecone_result
            },
            status_code=200
        )
        
    except Exception as e:
        error_msg = f"Error processing PDF content: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )
