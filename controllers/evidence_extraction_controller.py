from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import json
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from io import BytesIO
from pathlib import Path
import tempfile
import logging
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# Initialize OpenAI client
openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

router = APIRouter()

def check_file_exists_in_s3(s3_key: str) -> bool:
    """Check if a file exists in S3"""
    try:
        s3_client.head_object(
            Bucket=os.getenv('AWS_BUCKET_NAME'),
            Key=s3_key
        )
        return True
    except ClientError:
        return False

@router.post("/upload")
async def upload_file_for_evidence(file: UploadFile = File(...)):
    """Upload a PDF file and process it for evidence extraction"""
    try:
        logger.info(f"Starting upload for file: {file.filename}")
        
        # Read file into memory
        try:
            content = await file.read()
            logger.info(f"Successfully read file content, size: {len(content)} bytes")
        except Exception as e:
            logger.error(f"Error reading file content: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file content: {str(e)}"
            )

        base_name = Path(file.filename).stem
        pdf_key = f"evidence/{base_name}/{file.filename}"
        logger.info(f"Generated S3 key: {pdf_key}")

        # Check if file already exists
        try:
            if check_file_exists_in_s3(pdf_key):
                logger.warning(f"File already exists in S3: {pdf_key}")
                raise HTTPException(
                    status_code=400,
                    detail="A file with this name already exists"
                )
        except ClientError as e:
            logger.error(f"Error checking S3: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error checking S3: {str(e)}"
            )

        # Upload to S3
        try:
            s3_client.upload_fileobj(
                BytesIO(content),
                os.getenv('AWS_BUCKET_NAME'),
                pdf_key
            )
            logger.info(f"Successfully uploaded file to S3: {pdf_key}")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading to S3: {str(e)}"
            )
        
        # Create response in the expected JsonData format
        try:
            evidence_data = {
                "pages": [
                    {
                        "page": 1,
                        "text": "Sample evidence 1 from document",
                        "md": "",
                        "images": [],
                        "charts": [],
                        "items": [
                            {
                                "type": "text",
                                "value": "Sample evidence 1 from document",
                                "md": "",
                                "bBox": {
                                    "x": 0,
                                    "y": 0,
                                    "w": 0,
                                    "h": 0
                                }
                            }
                        ],
                        "status": "completed",
                        "links": [],
                        "width": 0,
                        "height": 0,
                        "triggeredAutoMode": False,
                        "structuredData": None,
                        "noStructuredContent": False,
                        "noTextContent": False
                    },
                    {
                        "page": 2,
                        "text": "Sample evidence 2 from document",
                        "md": "",
                        "images": [],
                        "charts": [],
                        "items": [
                            {
                                "type": "text",
                                "value": "Sample evidence 2 from document",
                                "md": "",
                                "bBox": {
                                    "x": 0,
                                    "y": 0,
                                    "w": 0,
                                    "h": 0
                                }
                            }
                        ],
                        "status": "completed",
                        "links": [],
                        "width": 0,
                        "height": 0,
                        "triggeredAutoMode": False,
                        "structuredData": None,
                        "noStructuredContent": False,
                        "noTextContent": False
                    }
                ],
                "job_metadata": {
                    "credits_used": 0,
                    "job_credits_usage": 0,
                    "job_pages": 2,
                    "job_auto_mode_triggered_pages": 0,
                    "job_is_cache_hit": False,
                    "credits_max": 0
                },
                "job_id": f"job_{base_name}",
                "file_path": pdf_key
            }

            bucket_name = os.getenv('AWS_BUCKET_NAME')
            region = os.getenv('AWS_REGION')
            if not bucket_name or not region:
                raise ValueError("AWS_BUCKET_NAME or AWS_REGION environment variables not set")

            pdf_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{pdf_key}"
            logger.info(f"Generated PDF URL: {pdf_url}")

            return JSONResponse(
                status_code=200,
                content={
                    "message": "File uploaded successfully",
                    "url": pdf_url,
                    "evidence": evidence_data
                }
            )
        except Exception as e:
            logger.error(f"Error preparing response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error preparing response: {str(e)}"
            )

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in upload_file_for_evidence: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.get("/{filename}")
async def get_evidence(filename: str):
    """Get extracted evidence for a specific file"""
    try:
        # TODO: Implement actual evidence retrieval
        # This is a placeholder that returns mock data
        evidence = {
            "sections": [
                {
                    "title": "Key Evidence",
                    "items": [
                        {
                            "text": f"Sample evidence 1 from {filename}",
                            "page": 1,
                            "confidence": 0.95
                        },
                        {
                            "text": f"Sample evidence 2 from {filename}",
                            "page": 2,
                            "confidence": 0.88
                        }
                    ]
                }
            ]
        }
        
        return JSONResponse(
            status_code=200,
            content=evidence
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evidence: {str(e)}"
        )

class ExtractionRequest(BaseModel):
    research_paper: str
    essay_topic: str
    temperature: float = 0
    max_tokens: int = 4000  # Reduced from 8192 to stay within Claude-3-Sonnet's limits

class ImageInfo(BaseModel):
    name: str
    height: int
    width: int
    x: int
    y: int
    original_width: int
    original_height: int
    type: str

class BBox(BaseModel):
    x: int
    y: int
    w: float
    h: float

class TextItem(BaseModel):
    type: str
    value: str
    md: str
    bBox: BBox

class Page(BaseModel):
    page: int
    text: str
    md: str = ""  # Make optional with default
    images: List[ImageInfo] = []  # Make optional with default empty list
    charts: List[Any] = []  # Make optional with default empty list
    items: List[TextItem] = []  # Make optional with default empty list
    status: str = "completed"  # Make optional with default
    links: List[Any] = []  # Make optional with default empty list
    width: int = 0  # Make optional with default
    height: int = 0  # Make optional with default
    triggeredAutoMode: bool = False  # Make optional with default
    structuredData: Optional[Any] = None  # Make optional
    noStructuredContent: bool = False  # Make optional with default
    noTextContent: bool = False  # Make optional with default

class JobMetadata(BaseModel):
    credits_used: float = 0  # Make optional with default
    job_credits_usage: int = 0  # Make optional with default
    job_pages: int = 0  # Make optional with default
    job_auto_mode_triggered_pages: int = 0  # Make optional with default
    job_is_cache_hit: bool = False  # Make optional with default
    credits_max: int = 0  # Make optional with default

class JsonData(BaseModel):
    pages: List[Page]
    job_metadata: JobMetadata = JobMetadata()  # Make optional with default
    job_id: str = "default_job_id"  # Make optional with default
    file_path: str = ""  # Make optional with default

class ProcessEvidenceRequest(BaseModel):
    file_name: str
    json_data: JsonData
    essay_topic: str = "Analyze how the main character demonstrates willpower and resilience throughout their journey"

class Evidence(BaseModel):
    raw_text: str
    meaning: str
    relevance_score: float

class PaperAnalysis(BaseModel):
    summary: str
    methodology: str
    key_findings: List[str]
    relevance_to_topic: str
    themes: List[Dict[str, str]]

class ExtractionResponse(BaseModel):
    extractions: List[Evidence]
    analysis: PaperAnalysis

@router.post("/raw-extract")
async def extract_raw_evidence(request: ProcessEvidenceRequest):
    """Extract raw evidence from a research paper using GPT-4"""
    try:
        # Extract text from JSON first
        research_paper = extract_text_from_json(request.json_data)
        logger.info(f"Processing evidence extraction for file: {request.file_name}")
        
        # Create the chat completion with JSON mode
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert research analyst tasked with extracting key insights from research papers.
Your responses must be in JSON format with the following structure:
{
    "extractions": [
        {
            "raw_text": "exact quote from paper",
            "meaning": "explanation of relevance",
            "relevance_score": float between 0 and 1
        }
    ],
    "analysis": {
        "summary": "brief summary",
        "methodology": "research methodology",
        "key_findings": ["finding1", "finding2", ...],
        "relevance_to_topic": "explanation of relevance",
        "themes": [
            {
                "theme": "theme name",
                "relevance": "theme relevance"
            }
        ]
    }
}"""
                },
                {
                    "role": "user",
                    "content": f"""Analyze the following research paper and extract evidence relevant to the essay topic.

Research Paper:
{research_paper}

Essay Topic:
{request.essay_topic}

Provide a thorough analysis and extract 3-5 key pieces of evidence that support or relate to the essay topic. 
Ensure each extraction includes the exact text from the paper and a detailed explanation of its relevance."""
                }
            ],
            temperature=0.7,
            max_tokens=4000
        )

        # Log the response
        logger.info(f"Completed evidence extraction for file: {request.file_name}")
        
        try:
            # Parse the JSON response
            content = json.loads(response.choices[0].message.content)
            
            # Validate against our Pydantic model
            validated_content = ExtractionResponse(**content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse model response as JSON"
            )
        except ValidationError as e:
            logger.error(f"Response validation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Model response did not match expected schema"
            )

        # Store the result in S3
        result = {
            "timestamp": datetime.now().isoformat(),
            "model": response.model,
            "content": validated_content.model_dump(),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

        # Generate S3 key for the result
        result_key = f"evidence/{Path(request.file_name).stem}/extraction_result.json"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=os.getenv('AWS_BUCKET_NAME'),
            Key=result_key,
            Body=json.dumps(result, indent=2)
        )

        return JSONResponse(
            status_code=200,
            content={
                "message": "Evidence extracted successfully",
                "result": validated_content.model_dump(),
                "metadata": {
                    "model": response.model,
                    "usage": result["usage"]
                }
            }
        )

    except Exception as e:
        logger.error(f"Error in evidence extraction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing evidence extraction: {str(e)}"
        )

@router.post("/process-evidence")
async def process_evidence(request: ProcessEvidenceRequest):
    """Process the parsed JSON and extract evidence using GPT-4"""
    try:
        print("\n=== Processing Evidence ===")
        print("Request data:", request.model_dump())
        print("File name:", request.file_name)
        print("JSON data:", request.json_data.model_dump())
        print("Essay topic:", request.essay_topic)
        
        # Extract and clean the text directly from the provided JSON
        try:
            result = await extract_raw_evidence(request)
            
            # Since result is a JSONResponse, we need to access its content
            result_data = result.body.decode()  # Convert bytes to string
            result_json = json.loads(result_data)  # Parse JSON string
            
            print("\nGPT's Response:")
            print("=" * 40)
            print(json.dumps(result_json["result"], indent=2))  # Print the result part which contains our extractions
            print("=" * 40)
            
            return result  # Return the original JSONResponse
            
        except Exception as inner_e:
            print(f"Inner error: {str(inner_e)}")
            print(f"Inner error type: {type(inner_e)}")
            import traceback
            print(f"Inner traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process evidence: {str(inner_e)}"
            )
            
    except Exception as e:
        print(f"\nOuter error in process_evidence: {str(e)}")
        print(f"Outer error type: {type(e)}")
        import traceback
        print(f"Outer traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process evidence: {str(e)}"
        )

def extract_text_from_json(json_data: JsonData):
    """Extract and clean text from the parsed JSON data"""
    try:
        print("\n=== Extracting Text from JSON ===")
        print("JSON Data Type:", type(json_data))
        
        # Get pages data
        pages_data = json_data.pages
        if not pages_data:
            raise ValueError("No pages found in JSON data")
            
        all_text = []
        print(f"\nProcessing {len(pages_data)} pages...")
        
        # Iterate through all pages
        for page_data in pages_data:
            # Get the clean text from the page
            page_text = page_data.text.strip()
            if page_text:
                print(f"\nPage {page_data.page} text:")
                print("-" * 40)
                print(page_text[:200] + "..." if len(page_text) > 200 else page_text)  # Show preview
                print("-" * 40)
                
                # Clean the text before adding
                # 1. Remove excessive whitespace
                cleaned_page = re.sub(r'\s+', ' ', page_text)
                # 2. Fix spacing around punctuation
                cleaned_page = re.sub(r'\s+([.,!?])', r'\1', cleaned_page)
                # 3. Ensure proper sentence spacing
                cleaned_page = re.sub(r'([.!?])\s*', r'\1\n\n', cleaned_page)
                
                all_text.append(cleaned_page)
        
        # Join all text with proper spacing
        print("\nCombining all pages...")
        combined_text = "\n\n".join(all_text)
        
        # Final cleaning
        print("\nFinal cleaning...")
        # 1. Normalize quotes
        cleaned_text = re.sub(r'["""]', '"', combined_text)
        # 2. Remove any remaining multiple newlines
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        # 3. Ensure proper spacing after newlines
        cleaned_text = re.sub(r'\n\s+', '\n', cleaned_text)
        
        final_text = cleaned_text.strip()
        
        print("\nFinal cleaned text preview:")
        print("=" * 40)
        print(final_text[:500] + "..." if len(final_text) > 500 else final_text)
        print("=" * 40)
        print(f"Total length: {len(final_text)} characters")
        
        return final_text
        
    except Exception as e:
        print(f"Error in extract_text_from_json: {str(e)}")
        print("JSON data structure:", json_data)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

@router.delete("/delete/{filename}")
async def delete_file(filename: str):
    """Delete a file from S3 and any associated evidence"""
    try:
        logger.info(f"Starting delete operation for file: {filename}")
        
        # Check environment variables
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        if not bucket_name:
            logger.error("AWS_BUCKET_NAME environment variable not set")
            raise HTTPException(
                status_code=500,
                detail="AWS configuration error: bucket name not set"
            )
        
        # Construct the S3 key for the evidence folder
        base_name = Path(filename).stem
        evidence_key = f"evidence/{base_name}/"
        logger.info(f"Looking for files with prefix: {evidence_key} in bucket: {bucket_name}")
        
        try:
            # List all objects with this prefix
            objects_to_delete = []
            paginator = s3_client.get_paginator('list_objects_v2')
            
            logger.info("Starting S3 object listing")
            page_count = 0
            for page in paginator.paginate(
                Bucket=bucket_name,
                Prefix=evidence_key
            ):
                page_count += 1
                if 'Contents' in page:
                    page_objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    objects_to_delete.extend(page_objects)
                    logger.info(f"Found {len(page_objects)} objects in page {page_count}")
                else:
                    logger.info(f"No objects found in page {page_count}")

            if not objects_to_delete:
                logger.warning(f"No files found in S3 with prefix: {evidence_key}")
                return JSONResponse(
                    status_code=404,
                    content={
                        "message": f"No files found for {filename}",
                        "details": {
                            "bucket": bucket_name,
                            "prefix": evidence_key
                        }
                    }
                )

            logger.info(f"Attempting to delete {len(objects_to_delete)} objects")
            logger.debug(f"Objects to delete: {objects_to_delete}")

            # Delete all objects with this prefix
            delete_response = s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': objects_to_delete}
            )
            
            # Log deletion results
            deleted = len(delete_response.get('Deleted', []))
            errors = delete_response.get('Errors', [])
            
            if errors:
                logger.error(f"Encountered errors while deleting: {errors}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Partial deletion failure",
                        "errors": errors,
                        "deleted_count": deleted
                    }
                )
            
            logger.info(f"Successfully deleted {deleted} files from S3")
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Successfully deleted {filename} and associated files",
                    "details": {
                        "deleted_count": deleted,
                        "files": [obj['Key'] for obj in objects_to_delete]
                    }
                }
            )

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS S3 error: {error_code} - {error_message}")
            logger.error(f"Full error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Error deleting from S3",
                    "error_code": error_code,
                    "error_message": error_message
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to delete file",
                "error": str(e),
                "type": type(e).__name__
            }
        )
