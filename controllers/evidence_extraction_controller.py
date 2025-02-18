"""
Evidence Extraction Controller

This module handles the extraction and processing of evidence from PDF documents.
It provides endpoints for uploading, parsing, and analyzing PDF files using GPT-4.

Key Features:
- PDF file upload and S3 storage
- Document parsing and text extraction
- Evidence analysis using GPT-4
- File management and cleanup

Environment Variables Required:
- AWS_ACCESS_KEY_ID: AWS access key
- AWS_SECRET_ACCESS_KEY: AWS secret key
- AWS_REGION: AWS region
- AWS_BUCKET_NAME: S3 bucket name
- OPENAI_API_KEY: OpenAI API key
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from pathlib import Path
from io import BytesIO
from botocore.exceptions import ClientError
import logging
from typing import Optional, List, Dict, Any
import boto3
from openai import OpenAI
import json
import re
from datetime import datetime
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import nest_asyncio
import s3fs
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
import shutil
from helpers.evidence_extraction_helper import process_raw_evidence

# Apply nest_asyncio to allow nested event loops (required for async operations)
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
logger = logging.getLogger(__name__)

# Initialize FastAPI router with evidence tag for API documentation
router = APIRouter(tags=["evidence"])

# Initialize AWS S3 client with credentials from environment
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )
    
    # Verify AWS credentials and bucket access
    try:
        s3_client.head_bucket(Bucket=os.getenv('AWS_BUCKET_NAME'))
        logger.info(f"Successfully connected to bucket: {os.getenv('AWS_BUCKET_NAME')}")
    except Exception as e:
        logger.error(f"Failed to access bucket: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to access S3 bucket: {str(e)}"
        )
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {str(e)}")
    raise HTTPException(
        status_code=500,
        detail=f"Failed to initialize S3 client: {str(e)}"
    )

# Initialize OpenAI client for GPT-4 integration
openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)


# Type Definitions
class ImageInfo(BaseModel):
    """Information about images extracted from documents.
    
    Attributes:
        name: Image filename or identifier
        height: Height of the image in pixels
        width: Width of the image in pixels
        x: X-coordinate position in the document
        y: Y-coordinate position in the document
        original_width: Original width before any scaling
        original_height: Original height before any scaling
        type: Image type/format (e.g., 'png', 'jpeg')
    """
    name: str
    height: int
    width: int
    x: int
    y: int
    original_width: int
    original_height: int
    type: str

class BBox(BaseModel):
    """Bounding box coordinates for text elements.
    
    Attributes:
        x: X-coordinate of the top-left corner
        y: Y-coordinate of the top-left corner
        w: Width of the bounding box
        h: Height of the bounding box
    """
    x: int
    y: int
    w: float
    h: float

class TextItem(BaseModel):
    """Text item extracted from the document.
    
    Attributes:
        type: Type of text (e.g., 'paragraph', 'heading')
        value: The actual text content
        md: Markdown formatted version of the text
        bBox: Bounding box coordinates of the text
    """
    type: str
    value: str
    md: str
    bBox: BBox

class Page(BaseModel):
    """Represents a page from the parsed document.
    
    Attributes:
        page: Page number
        text: Raw text content of the page
        md: Markdown formatted text content
        images: List of images on the page
        charts: List of charts/figures detected
        items: List of text items with positioning
        status: Processing status of the page
        links: List of hyperlinks found
        width: Page width in pixels
        height: Page height in pixels
        triggeredAutoMode: Whether auto-processing was triggered
        structuredData: Additional structured data extracted
        noStructuredContent: Flag indicating no structured content
        noTextContent: Flag indicating no text content
    """
    page: int
    text: str
    md: str = ""
    images: List[ImageInfo] = []
    charts: List[Any] = []
    items: List[TextItem] = []
    status: str = "completed"
    links: List[Any] = []
    width: int = 0
    height: int = 0
    triggeredAutoMode: bool = False
    structuredData: Optional[Any] = None
    noStructuredContent: bool = False
    noTextContent: bool = False

class JobMetadata(BaseModel):
    """Metadata about the document processing job.
    
    Attributes:
        credits_used: Number of API credits consumed
        job_credits_usage: Total credits allocated
        job_pages: Number of pages processed
        job_auto_mode_triggered_pages: Pages that triggered auto-mode
        job_is_cache_hit: Whether results were from cache
        credits_max: Maximum credits allowed
    """
    credits_used: float = 0
    job_credits_usage: int = 0
    job_pages: int = 0
    job_auto_mode_triggered_pages: int = 0
    job_is_cache_hit: bool = False
    credits_max: int = 0

class JsonData(BaseModel):
    """Complete JSON data structure for parsed documents.
    
    Attributes:
        pages: List of parsed pages
        job_metadata: Processing job metadata
        job_id: Unique identifier for the job
        file_path: Path to the processed file
    """
    pages: List[Page]
    job_metadata: JobMetadata = JobMetadata()
    job_id: str = "default_job_id"
    file_path: str = ""

class EvidenceReasoning(BaseModel):
    """Detailed reasoning for a piece of evidence.
    
    Attributes:
        specific_relevance: How the evidence relates to the topic
        application: Where this evidence can be applied
        insights: Detailed insights from the evidence
    """
    specific_relevance: str
    application: str
    insights: str

class EvidenceStrength(BaseModel):
    """Assessment of evidence strength.
    
    Attributes:
        score: Strength score (high, moderate, low)
        justification: Justification for the score
    """
    score: str
    justification: str

class Evidence(BaseModel):
    """A piece of evidence with detailed analysis."""
    exact_text: str
    category: str
    simplified_reasoning: str
    strength_of_evidence: EvidenceStrength

class ExtractionResponse(BaseModel):
    """Response format for evidence analysis."""
    refined_topic: str
    verified_evidence: List[Evidence]

class ProcessEvidenceRequest(BaseModel):
    """Request format for evidence processing.
    
    Attributes:
        file_name: Name of the file to process
        essay_topic: Topic to analyze evidence against
    """
    file_name: str
    essay_topic: str = "Why gaming is great and should be encouraged?"

def check_file_exists_in_s3(key: str) -> bool:
    """Check if a file exists in S3"""
    try:
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise

@router.post("/upload")
async def upload_file_for_evidence(file: UploadFile = File(...)):
    """Upload a PDF file and process it for evidence extraction"""
    try:
        logger.info(f"Starting upload for file: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )
        
        # Read file into memory
        try:
            content = await file.read()
            logger.info(f"Successfully read file content, size: {len(content)} bytes")
        except Exception as e:
            logger.info(f"Error reading file content: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file content: {str(e)}"
            )

        # Get bucket name
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        if not bucket_name:
            logger.info("AWS_BUCKET_NAME environment variable not set")
            raise HTTPException(
                status_code=500,
                detail="AWS configuration error: bucket name not set"
            )

        # Generate S3 key - store in folder with same name as file
        base_name = Path(file.filename).stem
        pdf_key = f"documents/{base_name}/original_document/{file.filename}"
        logger.info(f"Generated S3 key: {pdf_key}")

        # Check if file already exists
        try:
            if check_file_exists_in_s3(pdf_key):
                logger.info(f"File already exists in S3: {pdf_key}")
                raise HTTPException(
                    status_code=400,
                    detail="A file with this name already exists"
                )
        except ClientError as e:
            logger.info(f"Error checking S3: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error checking S3: {str(e)}"
            )

        # Upload to S3
        try:
            s3_client.upload_fileobj(
                BytesIO(content),
                bucket_name,
                pdf_key,
                ExtraArgs={
                    'ContentType': 'application/pdf'
                }
            )
            logger.info(f"Successfully uploaded file to S3: {pdf_key}")

            # Generate a pre-signed URL for the uploaded file
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': pdf_key
                },
                ExpiresIn=3600  # URL expires in 1 hour
            )

            return JSONResponse(
                status_code=200,
                content={
                    "message": "File uploaded successfully",
                    "file": {
                        "name": file.filename,
                        "size": len(content),
                        "url": url
                    }
                }
            )

        except Exception as e:
            logger.info(f"Error uploading file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading file: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.info(f"Unexpected error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )

@router.get("/list")
async def list_files_endpoint():
    """List all files endpoint with PDF details from original_document folders"""
    try:
        # First, get all base directories
        base_response = s3_client.list_objects_v2(
            Bucket=os.getenv('AWS_BUCKET_NAME'),
            Prefix='documents/',
            Delimiter='/'
        )
        
        files = []
        # Process each base directory
        if 'CommonPrefixes' in base_response:
            for prefix in base_response['CommonPrefixes']:
                base_dir = prefix['Prefix']
                # Look for PDF in original_document subfolder
                pdf_response = s3_client.list_objects_v2(
                    Bucket=os.getenv('AWS_BUCKET_NAME'),
                    Prefix=f"{base_dir}original_document/"
                )
                
                if 'Contents' in pdf_response:
                    for item in pdf_response['Contents']:
                        if item['Key'].lower().endswith('.pdf'):
                            files.append({
                                'folder': base_dir.replace('documents/', '').rstrip('/'),
                                'pdf_name': os.path.basename(item['Key']),
                                'size': item['Size'],
                                'last_modified': item['LastModified'].isoformat()
                            })
        
        return JSONResponse(content={'files': files})
        
    except ClientError as e:
        logger.info(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.info(f"Unexpected error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parse/{filename}")
async def parse_pdf(filename: str) -> Dict[str, Any]:
    """Parse a PDF file and extract evidence from it"""
    logger.info(f"Received parse request for file: {filename}")
    
    try:
        # Construct the S3 key
        base_name = Path(filename).stem
        pdf_key = f"documents/{base_name}/original_document/{filename}"
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        
        # Check if file exists in S3
        try:
            s3_client.head_object(Bucket=bucket_name, Key=pdf_key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise HTTPException(
                    status_code=404,
                    detail=f"File {filename} not found in S3 bucket"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error checking file in S3: {str(e)}"
                )
        
        # If we get here, file exists
        logger.info(f"File {filename} found successfully in S3")
        
        # Create an S3FileSystem instance for document parsing
        fs = s3fs.S3FileSystem()
        
        # List files in the S3 prefix
        s3_prefix = f"documents/{base_name}/original_document"
        files = fs.ls(f"s3://{bucket_name}/{s3_prefix}")
        logger.info(f"Found files in S3: {files}")
        
        try:
            # Create a temporary local directory for downloading
            temp_dir = "./temp_docs"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Clean existing files in temp directory
            for temp_file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, temp_file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.info(f"Failed to delete {file_path}. Reason: {str(e)}")
            
            # Download the file locally
            local_path = os.path.join(temp_dir, filename)
            fs.get(f"s3://{bucket_name}/{pdf_key}", local_path)
            
            # Initialize LlamaParse
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="text",
                verbose=True,
                invalidate_cache=True
            )
            
            # Configure file extractor with LlamaParse
            file_extractor = {'.pdf': parser}
            
            # Load documents using SimpleDirectoryReader with LlamaParse extractor
            reader = SimpleDirectoryReader(temp_dir, file_extractor=file_extractor)
            documents = await reader.aload_data()
            logger.info(f"Loaded documents: {documents}")
            
            # Extract text content from documents
            document_texts = []
            for i, doc in enumerate(documents):
                logger.info(f"Document {i}:")
                logger.info(doc.text)
                document_texts.append(doc.text)
            
            # Prepare the parsed result JSON
            parsed_result = {
                "message": f"PDF file {filename} parsed successfully",
                "location": f"s3://{bucket_name}/{pdf_key}",
                "exists": True,
                "document_count": len(documents),
                "documents": document_texts
            }
            
            # Upload parsed result to S3
            parsed_json_key = f"documents/{base_name}/parsed_json/{filename.replace('.pdf', '.json')}"
            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=parsed_json_key,
                    Body=json.dumps(parsed_result, indent=2),
                    ContentType='application/json'
                )
                logger.info(f"Uploaded parsed result to S3: {parsed_json_key}")
            except Exception as upload_error:
                logger.info(f"Error uploading parsed result to S3: {str(upload_error)}")
                # Continue even if upload fails, as we still want to return the parsed result
            
            # Clean up temporary files
            os.remove(local_path)
            os.rmdir(temp_dir)
            
            return parsed_result
            
        except Exception as parse_error:
            logger.info(f"Error parsing documents: {str(parse_error)}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing documents: {str(parse_error)}"
            )

    except Exception as e:
        logger.info(f"Error in parse_pdf: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.post("/raw-extract")
async def extract_raw_evidence(request: ProcessEvidenceRequest):
    """Extract raw evidence from a research paper using GPT-4"""
    return await process_raw_evidence(request.file_name, request.essay_topic, ExtractionResponse)


@router.get("/list-evidence", response_model=List[Dict])
async def list_evidence():
    """
    List and combine evidence extractions from all files in S3.
    Returns a list of evidence items with their metadata.
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        
        # List all document directories
        paginator = s3_client.get_paginator('list_objects_v2')
        all_extractions = []
        
        try:
            # First, get all document directories
            logger.info("\n" + "="*50)
            logger.info("STARTING EVIDENCE LISTING")
            logger.info("="*50 + "\n")
            
            for page in paginator.paginate(
                Bucket=os.getenv('AWS_BUCKET_NAME'),
                Prefix='documents/',
                Delimiter='/'  # Use delimiter to get directories
            ):
                # Process each document directory
                logger.info("\nFOUND DIRECTORIES:")
                for prefix in page.get('CommonPrefixes', []):
                    logger.info(f"* {prefix.get('Prefix')}")
                
                for prefix in page.get('CommonPrefixes', []):
                    doc_prefix = prefix.get('Prefix', '')  # e.g., "documents/Attention is all you need/"
                    logger.info(f"\n>>> Processing directory: {doc_prefix}")
                    
                    # Look for evidence files in the extracted_evidence directory
                    evidence_prefix = f"{doc_prefix}extracted_evidence/"
                    logger.info(f">>> Looking for evidence in: {evidence_prefix}")
                    
                    # List contents of the extracted_evidence directory
                    for evidence_page in paginator.paginate(
                        Bucket=os.getenv('AWS_BUCKET_NAME'),
                        Prefix=evidence_prefix
                    ):
                        if 'Contents' in evidence_page:
                            for obj in evidence_page['Contents']:
                                logger.info(f">>> Found file: {obj['Key']}")
                                if obj['Key'].endswith('_evidence.json'):
                                    logger.info(f"\n!!! Processing evidence file: {obj['Key']}")
                                    try:
                                        # Get the file content
                                        response = s3_client.get_object(
                                            Bucket=os.getenv('AWS_BUCKET_NAME'),
                                            Key=obj['Key']
                                        )
                                        content = json.loads(response['Body'].read().decode('utf-8'))
                                        logger.info(f"!!! Successfully read content from {obj['Key']}")
                                        
                                        # Extract document name (folder name)
                                        doc_name = doc_prefix.rstrip('/').split('/')[-1]
                                        file_name = obj['Key'].split('/')[-1]
                                        logger.info(f"!!! Document: {doc_name}, File: {file_name}")
                                        
                                        if 'analysis' in content:
                                            # Add metadata to each extraction
                                            for extraction in content['analysis'].get('verified_evidence', []):
                                                enriched_extraction = {
                                                    'document_name': doc_name,
                                                    'file_name': file_name,
                                                    'essay_topic': content.get('essay_topic', ''),
                                                    'refined_topic': content['analysis'].get('refined_topic', ''),
                                                    'raw_text': extraction['exact_text'],
                                                    'category': extraction['category'],
                                                    'reasoning': extraction['simplified_reasoning'],
                                                    'strength': extraction['strength_of_evidence']['score'],
                                                    'strength_justification': extraction['strength_of_evidence']['justification']
                                                }
                                                logger.info(f"Enriched extraction: {enriched_extraction}")
                                                all_extractions.append(enriched_extraction)
                                        
                                    except Exception as e:
                                        logger.info(f"Error processing file {obj['Key']}: {str(e)}")
                                        continue
        
        except Exception as e:
            logger.info(f"Error listing evidence files: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list evidence files: {str(e)}"
            )
        
        # Sort extractions by strength in descending order
        strength_order = {'High': 3, 'Moderate': 2, 'Low': 1}
        all_extractions.sort(key=lambda x: strength_order.get(x['strength'], 0), reverse=True)
        return all_extractions
    
    except Exception as e:
        logger.info(f"Failed to list evidence files: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list evidence files: {str(e)}"
        )

@router.delete("/delete/{filename}")
async def delete_file(filename: str):
    """Delete an entire document directory from S3, including all subdirectories and files"""
    try:
        logger.info(f"Starting deletion of document directory for file: {filename}")
        
        # Validate AWS bucket configuration
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        if not bucket_name:
            logger.info("AWS_BUCKET_NAME environment variable not set")
            raise HTTPException(
                status_code=500,
                detail="AWS configuration error: bucket name not set"
            )
        
        # Construct the base directory path for the document
        document_name = Path(filename).stem
        document_directory = f"documents/{document_name}/"
        logger.info(f"Preparing to delete directory: {document_directory} from bucket: {bucket_name}")
        
        try:
            # Initialize list for tracking all objects in the directory
            objects_for_deletion = []
            s3_paginator = s3_client.get_paginator('list_objects_v2')
            
            # List all objects in the document directory (including subdirectories)
            logger.info("Scanning S3 for all files in document directory")
            page_number = 0
            for page in s3_paginator.paginate(
                Bucket=bucket_name,
                Prefix=document_directory
            ):
                page_number += 1
                if 'Contents' in page:
                    current_page_objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    objects_for_deletion.extend(current_page_objects)
                    logger.info(f"Found {len(current_page_objects)} files in page {page_number}")
                else:
                    logger.info(f"No files found in page {page_number}")

            # Check if any files were found
            if not objects_for_deletion:
                logger.info(f"No files found in directory: {document_directory}")
                return JSONResponse(
                    status_code=404,
                    content={
                        "message": f"No files found for document: {document_name}",
                        "details": {
                            "bucket": bucket_name,
                            "directory": document_directory
                        }
                    }
                )

            logger.info(f"Initiating deletion of {len(objects_for_deletion)} files")
            logger.info(f"Files to be deleted: {objects_for_deletion}")

            # Delete all files in the directory
            deletion_response = s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': objects_for_deletion}
            )
            
            # Process deletion results
            successfully_deleted = len(deletion_response.get('Deleted', []))
            deletion_errors = deletion_response.get('Errors', [])
            
            # Handle any deletion errors
            if deletion_errors:
                logger.info(f"Errors occurred during deletion: {deletion_errors}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Some files could not be deleted",
                        "errors": deletion_errors,
                        "successfully_deleted_count": successfully_deleted
                    }
                )
            
            logger.info(f"Successfully deleted {successfully_deleted} files from document directory")
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Successfully deleted document directory for {filename}",
                    "details": {
                        "files_deleted": successfully_deleted,
                        "directory_path": document_directory,
                        "deleted_files": [obj['Key'] for obj in objects_for_deletion]
                    }
                }
            )

        except ClientError as s3_error:
            error_code = s3_error.response['Error']['Code']
            error_message = s3_error.response['Error']['Message']
            logger.info(f"AWS S3 error: {error_code} - {error_message}")
            logger.info(f"Full error details: {str(s3_error)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Error accessing S3",
                    "error_code": error_code,
                    "error_details": error_message
                }
            )

    except HTTPException:
        raise
    except Exception as unexpected_error:
        logger.info(f"Unexpected error during deletion: {str(unexpected_error)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Unexpected error during file deletion",
                "error": str(unexpected_error),
                "error_type": type(unexpected_error).__name__
            }
        )

def extract_text_from_json(json_data: JsonData):
    """Extract and clean text from the parsed JSON data"""
    try:
        logger.info("\n=== Extracting Text from JSON ===")
        logger.info("JSON Data Type:", type(json_data))
        
        # Get pages data
        pages_data = json_data.pages
        if not pages_data:
            raise ValueError("No pages found in JSON data")
            
        all_text = []
        logger.info(f"\nProcessing {len(pages_data)} pages...")
        
        # Iterate through all pages
        for page_data in pages_data:
            # Get the clean text from the page
            page_text = page_data.text.strip()
            if page_text:
                logger.info(f"\nPage {page_data.page} text:")
                logger.info("-" * 40)
                logger.info(page_text[:200] + "..." if len(page_text) > 200 else page_text)  # Show preview
                logger.info("-" * 40)
                
                # Clean the text before adding
                # 1. Remove excessive whitespace
                cleaned_page = re.sub(r'\s+', ' ', page_text)
                # 2. Fix spacing around punctuation
                cleaned_page = re.sub(r'\s+([.,!?])', r'\1', cleaned_page)
                # 3. Ensure proper sentence spacing
                cleaned_page = re.sub(r'([.!?])\s*', r'\1\n\n', cleaned_page)
                
                all_text.append(cleaned_page)
        
        # Join all text with proper spacing
        logger.info("\nCombining all pages...")
        combined_text = "\n\n".join(all_text)
        
        # Final cleaning
        logger.info("\nFinal cleaning...")
        # 1. Normalize quotes
        cleaned_text = re.sub(r'["""]', '"', combined_text)
        # 2. Remove any remaining multiple newlines
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        # 3. Ensure proper spacing after newlines
        cleaned_text = re.sub(r'\n\s+', '\n', cleaned_text)
        
        final_text = cleaned_text.strip()
        
        logger.info("\nFinal cleaned text preview:")
        logger.info("=" * 40)
        logger.info(final_text[:500] + "..." if len(final_text) > 500 else final_text)
        logger.info("=" * 40)
        logger.info(f"Total length: {len(final_text)} characters")
        
        return final_text
        
    except Exception as e:
        logger.info(f"Error in extract_text_from_json: {str(e)}")
        logger.info("JSON data structure:", json_data)
        import traceback
        logger.info(f"Traceback: {traceback.format_exc()}")
        raise
