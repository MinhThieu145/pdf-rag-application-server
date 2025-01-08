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
import anthropic
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import re
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize AWS S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY')
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

@router.post("/evidence/upload")
async def upload_file_for_evidence(file: UploadFile = File(...)):
    """Upload a PDF file and process it for evidence extraction"""
    try:
        # Read file into memory
        content = await file.read()
        base_name = Path(file.filename).stem
        pdf_key = f"evidence/{base_name}/{file.filename}"

        # Check if file already exists
        if check_file_exists_in_s3(pdf_key):
            raise HTTPException(
                status_code=400,
                detail="A file with this name already exists"
            )

        # Upload to S3
        s3_client.upload_fileobj(
            BytesIO(content),
            os.getenv('AWS_BUCKET_NAME'),
            pdf_key
        )
        
        # TODO: Add evidence extraction processing here
        # Placeholder for evidence extraction logic
        extracted_evidence = {
            "sections": [
                {
                    "title": "Key Evidence",
                    "items": [
                        {
                            "text": "Sample evidence 1 from document",
                            "page": 1,
                            "confidence": 0.95
                        },
                        {
                            "text": "Sample evidence 2 from document",
                            "page": 2,
                            "confidence": 0.88
                        }
                    ]
                }
            ]
        }

        pdf_url = f"https://{os.getenv('AWS_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{pdf_key}"
        return JSONResponse(
            status_code=200,
            content={
                "message": "File processed successfully",
                "pdf_url": pdf_url,
                "filename": file.filename,
                "evidence": extracted_evidence
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )

@router.get("/evidence/{filename}")
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

@router.post("/raw-extract")
async def extract_raw_evidence(request: ProcessEvidenceRequest):
    """Extract raw evidence from a research paper using Claude"""
    try:
        # Extract text from JSON first
        research_paper = extract_text_from_json(request.json_data)
        
        # Create extraction request with the proper type
        extraction_request = ExtractionRequest(
            research_paper=research_paper,
            essay_topic=request.essay_topic
        )
        
        # Create the message prompt
        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=extraction_request.max_tokens,
            temperature=extraction_request.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are an expert research analyst tasked with extracting key insights from a research paper to support an argumentative essay. Your goal is to provide detailed, relevant, and contextually rich evidence that can be used effectively in the essay.

First, carefully read the following research paper and essay topic:

<research_paper>
{research_paper}
</research_paper>

<essay_topic>
{request.essay_topic}
</essay_topic>

Your task is to extract 3-5 key insights from the research paper that can serve as strong evidence for the given essay topic. Follow these steps:

1. Analyze the paper thoroughly, keeping the essay topic in mind.
2. Identify passages that provide specific, detailed evidence related to the topic.
3. Extract verbatim quotes from these passages.
4. Ensure each extraction is complete and contains sufficient context to be understood without reading the entire paper.
5. Evaluate how each extraction can be used in the essay (e.g., supporting evidence, counterargument, context).

Before presenting your final extractions, show your thought process inside <paper_analysis_and_extraction> tags. This section should be thorough and may be quite long. Include the following steps:

1. Briefly summarize the main points of the research paper, its methodology, and key findings.
2. List out the key claims or arguments from the essay topic.
3. Provide a brief evaluation of the paper's overall relevance to the essay topic.
4. Identify and number key themes or arguments in the paper.
5. For each theme, write down relevant quotes, numbering them and noting their potential relevance to the essay topic.
6. For each quote:
   a. Copy the verbatim quote.
   b. Check if it provides enough context. If not, expand the quote to include necessary information.
   c. Classify the quote as supporting the essay topic, opposing it, or providing neutral context.
   d. Write brief arguments for and against including this quote in the final selection.
   e. Consider how it could be used in the essay, including potential counterarguments.
   f. Clearly mark the start and end of each extraction using <start> and <end> tags.
7. Select the 3-5 strongest extractions based on your analysis, ensuring a balance of supporting, opposing, and contextual evidence if possible.

Present your final extractions in the following format:

<extraction n>
raw text: <start>[Insert the verbatim quote from the research paper here]</start><end>[If needed, insert another part of the quote here]</end>
meaning: [Explain how this extraction can be used in the essay. Discuss its relevance to the topic and how it can support arguments or counterarguments.]
</extraction n>

Where 'n' is the number of the extraction (1, 2, 3, etc.).

Begin your response with "Here are the key extractions from the research paper relevant to the essay topic:" and then proceed with your numbered extractions."""
                        }
                    ]
                }
            ]
        )

        # Print Claude's response
        print("\n=== Claude's Analysis ===")
        content = message.content[0].text if message.content else ""
        print(content)
        print("=======================\n")

        # Store the result in S3
        result = {
            "timestamp": datetime.now().isoformat(),
            "model": message.model,
            "content": content,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens
            }
        }

        # Generate a unique key for this extraction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_key = f"evidence/raw_extractions/{timestamp}.json"

        # Upload to S3
        s3_client.put_object(
            Bucket=os.getenv('AWS_BUCKET_NAME'),
            Key=result_key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )

        # Generate URL for the result
        result_url = f"https://{os.getenv('AWS_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{result_key}"

        return JSONResponse(
            status_code=200,
            content={
                "message": "Evidence extracted successfully",
                "result_url": result_url,
                "extraction": result
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract evidence: {str(e)}"
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

@router.post("/process-evidence")
async def process_evidence(request: ProcessEvidenceRequest):
    """Process the parsed JSON and extract evidence using Claude"""
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
            
            print("\nClaude's Response:")
            print("=" * 40)
            print(result_json["extraction"]["content"])
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
