from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from io import BytesIO
from pathlib import Path
import tempfile
from llama_parse import LlamaParse
import nest_asyncio
import httpx

# Apply nest_asyncio to allow nested async loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize AWS S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Initialize parser
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    verbose=True
)

# Settings
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

@router.get("/list")
async def list_files():
    """List all PDF files in S3"""
    try:
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='documents/'
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.pdf'):
                    # Generate presigned URL for frontend access
                    url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': bucket_name,
                            'Key': obj['Key']
                        },
                        ExpiresIn=3600  # URL expires in 1 hour
                    )
                    
                    files.append({
                        'key': obj['Key'],
                        'name': os.path.basename(obj['Key']),
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'url': url
                    })
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Files retrieved successfully",
                "files": files
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {str(e)}"
        )

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF file to S3"""
    try:
        # Read file into memory
        content = await file.read()
        base_name = Path(file.filename).stem
        pdf_key = f"documents/{base_name}/{file.filename}"

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
        
        pdf_url = f"https://{os.getenv('AWS_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{pdf_key}"
        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "pdf_url": pdf_url,
                "filename": file.filename
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )

@router.post("/parse/{filename}")
async def parse_pdf(filename: str):
    """Parse a PDF file that's already in S3 and generate markdown"""
    temp_pdf_path = None
    try:
        print(f"Starting to parse PDF: {filename}")
        # Get file info
        clean_filename = filename.split('/')[-1]
        base_name = Path(clean_filename).stem
        s3_key = f"documents/{base_name}/{clean_filename}"
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        
        print(f"Looking for file in S3: {s3_key}")
        # Download from S3
        pdf_buffer = BytesIO()
        try:
            s3_client.download_fileobj(bucket_name, s3_key, pdf_buffer)
            pdf_buffer.seek(0)
            print(f"Successfully downloaded file, size: {len(pdf_buffer.getvalue())} bytes")
        except ClientError as e:
            print(f"S3 download error: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=f"File not found in S3: {s3_key}"
            )
        
        # Parse PDF
        print("Creating temporary file...")
        temp_pdf_fd, temp_pdf_path = tempfile.mkstemp(suffix='.pdf')
        try:
            os.close(temp_pdf_fd)
            print(f"Writing to temporary file: {temp_pdf_path}")
            with open(temp_pdf_path, 'wb') as f:
                f.write(pdf_buffer.getvalue())
            
            print("Initializing parser...")
            # Re-initialize parser for each request to ensure clean state
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                verbose=True,
                take_screenshot=True,
                exact_charts=True  
            )
            
            print("Starting PDF parsing...")
            try:
                json_result = parser.get_json_result(temp_pdf_path)
                print("Successfully parsed PDF")
                job_id = json_result[0]["job_id"]
                print(f"Got job ID: {job_id}")
            except Exception as parse_error:
                print(f"Parser error: {str(parse_error)}")
                print(f"Parser error type: {type(parse_error)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"PDF parsing failed: {str(parse_error)}"
                )
                
        finally:
            if os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                    print("Cleaned up temporary file")
                except Exception as e:
                    print(f"Warning: Could not delete temporary file: {str(e)}")
        
        # Upload result to S3
        print("Preparing to upload JSON result...")
        json_filename = f"{base_name}.json"
        json_key = f"documents/{base_name}/json/{json_filename}"
        try:
            json_str = json.dumps(json_result, ensure_ascii=False, indent=2)
            json_buffer = BytesIO(json_str.encode('utf-8'))
            
            s3_client.upload_fileobj(
                json_buffer,
                bucket_name,
                json_key
            )
            print(f"Successfully uploaded JSON to {json_key}")
        except Exception as upload_error:
            print(f"Error uploading JSON: {str(upload_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload parsing results: {str(upload_error)}"
            )
        
        json_url = f"https://{bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{json_key}"
        print("Returning successful response")
        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF parsed successfully",
                "json_url": json_url,
                "job_id": job_id
            }
        )

    except HTTPException as e:
        print(f"HTTP Exception: {str(e)}")
        raise e
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Error type: {type(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse PDF: {str(e)}"
        )

async def download_and_upload_screenshot_memory(job_id: str, image_name: str, pdf_name: str, page_number: int) -> str:
    """Download a page screenshot from LlamaIndex API and upload to S3 using memory buffers"""
    print(f"Downloading screenshot for job {job_id}, image {image_name}")
    url = f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/image/{image_name}"
    headers = {
        'Accept': 'image/jpeg',
        'Authorization': f'Bearer {os.getenv("LLAMA_CLOUD_API_KEY")}',
    }

    print(f"Making request to {url}")
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        print(f"Got response with status code: {response.status_code}")
        if response.status_code == 200:
            # Store image in memory buffer
            image_buffer = BytesIO(response.content)
            print(f"Downloaded image, size: {len(response.content)} bytes")
            
            # Create filename that ensures correct ordering
            base_name = Path(pdf_name).stem
            screenshot_name = f"page_{page_number:03d}.jpg"  # This ensures proper ordering: 001, 002, etc.
            
            # Upload directly from memory buffer
            bucket_name = os.getenv('AWS_BUCKET_NAME')
            s3_key = f"documents/{base_name}/screenshots/{screenshot_name}"
            print(f"Uploading to S3: {s3_key}")
            
            try:
                image_buffer.seek(0)
                s3_client.upload_fileobj(image_buffer, bucket_name, s3_key)
                url = f"https://{bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
                print(f"Successfully uploaded screenshot to {url}")
                return url
            except ClientError as e:
                print(f"Error uploading screenshot to S3: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload screenshot to S3: {str(e)}"
                )
        else:
            error_msg = f"Failed to download screenshot {image_name}. Status code: {response.status_code}"
            if response.content:
                error_msg += f", Response: {response.content.decode('utf-8', errors='ignore')}"
            print(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )

@router.post("/process-screenshots/{filename}")
async def process_screenshots(filename: str, job_id: str):
    """Process and upload screenshots for each page of a PDF"""
    try:
        print(f"Processing screenshots for {filename} with job_id {job_id}")
        # Clean up filename
        clean_filename = filename.split('/')[-1]
        base_name = Path(clean_filename).stem
        
        # Download the JSON result from S3
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        json_key = f"documents/{base_name}/json/{base_name}.json"
        json_buffer = BytesIO()
        
        try:
            print(f"Downloading JSON from S3: {json_key}")
            s3_client.download_fileobj(bucket_name, json_key, json_buffer)
            json_buffer.seek(0)
            json_content = json_buffer.read().decode('utf-8')
            print(f"JSON content length: {len(json_content)} bytes")
            json_result = json.loads(json_content)
            print("Successfully parsed JSON result")
            print(f"Found {len(json_result[0]['pages'])} pages in the JSON result")
        except ClientError as e:
            print(f"Error downloading JSON from S3: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=f"JSON result not found in S3: {json_key}"
            )
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            print(f"JSON content: {json_content[:1000]}...")  # Print first 1000 chars
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON format: {str(e)}"
            )
        
        # Process and upload screenshots
        screenshot_urls = []
        for page_idx, page in enumerate(json_result[0]["pages"]):
            print(f"Processing page {page_idx + 1}")
            print(f"Page keys: {list(page.keys())}")
            if "images" in page:
                print(f"Found {len(page['images'])} images in page {page_idx + 1}")
                for image in page["images"]:
                    print(f"Image info: {image}")
                    if image.get("type") == "full_page_screenshot":
                        print(f"Found screenshot for page {page_idx + 1}")
                        # Upload screenshot with page number for proper ordering
                        screenshot_url = await download_and_upload_screenshot_memory(
                            job_id,
                            image["name"],
                            clean_filename,
                            page_idx + 1
                        )
                        screenshot_urls.append({
                            "page": page_idx + 1,
                            "url": screenshot_url
                        })
                        print(f"Uploaded screenshot for page {page_idx + 1}")
            else:
                print(f"No images found in page {page_idx + 1}")

        # Sort by page number to ensure correct order
        screenshot_urls.sort(key=lambda x: x["page"])
        print(f"Processed {len(screenshot_urls)} screenshots")

        return JSONResponse(
            status_code=200,
            content={
                "message": "Screenshots processed successfully",
                "screenshots": screenshot_urls
            }
        )

    except HTTPException as e:
        print(f"HTTP Exception in process_screenshots: {str(e)}")
        raise e
    except Exception as e:
        print(f"Error in process_screenshots: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process screenshots: {str(e)}"
        )

@router.get("/screenshots/{pdf_name}")
async def get_screenshots(pdf_name: str):
    """Get all screenshots for a specific PDF"""
    try:
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        base_name = Path(pdf_name).stem
        screenshots_prefix = f"documents/{base_name}/screenshots/"
        
        # List all screenshots for this PDF
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=screenshots_prefix
        )
        
        screenshots = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith(('.png', '.jpg', '.jpeg')):
                    # Generate presigned URL for frontend access
                    url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': bucket_name,
                            'Key': obj['Key']
                        },
                        ExpiresIn=3600  # URL expires in 1 hour
                    )
                    
                    # Extract page number from filename
                    filename = os.path.basename(obj['Key'])
                    page_num = int(filename.split('_')[1].split('.')[0])
                    
                    screenshots.append({
                        'key': obj['Key'],
                        'url': url,
                        'page': page_num
                    })
        
        # Sort screenshots by page number
        screenshots.sort(key=lambda x: x['page'])
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Screenshots retrieved successfully",
                "screenshots": screenshots
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get screenshots: {str(e)}"
        )

@router.get("/content/{filename}")
async def get_json_content(filename: str):
    """Get the JSON content for a specific file"""
    try:
        # Construct the S3 key for the JSON file
        base_name = Path(filename).stem
        json_key = f"documents/{base_name}/json/{base_name}.json"
        
        # Get the object from S3
        try:
            response = s3_client.get_object(
                Bucket=os.getenv('AWS_BUCKET_NAME'),
                Key=json_key
            )
            json_content = json.loads(response['Body'].read().decode('utf-8'))
            return JSONResponse(
                status_code=200,
                content=json_content
            )
        except ClientError as e:
            raise HTTPException(
                status_code=404,
                detail=f"JSON file not found: {json_key}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get JSON content: {str(e)}"
        )

@router.get("/processed")
async def get_processed_pdfs():
    """Get list of PDFs that have been processed and stored in Pinecone"""
    try:
        print("Getting processed PDFs...")  # Debug log
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        processed_pdfs = []
        
        # List all objects in the documents prefix
        print(f"Listing objects in bucket: {bucket_name}")  # Debug log
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix='documents/')
        
        # Track unique PDF names that have been processed
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                # Check if this is a processed file (has text-split-llama-parse-default folder)
                if '/text-split-llama-parse-default/' in key:
                    # Extract PDF name from path
                    parts = key.split('/')
                    if len(parts) >= 4:  # documents/[pdf_name]/text-split.../[pdf_name].json
                        pdf_name = parts[1]  # Get the PDF name from path
                        if pdf_name not in processed_pdfs:
                            processed_pdfs.append(pdf_name)
                            print(f"Found processed PDF: {pdf_name}")  # Debug log
        
        # Get full details for each processed PDF
        pdf_details = []
        for pdf_name in processed_pdfs:
            try:
                # Get the original PDF metadata
                pdf_key = f"documents/{pdf_name}/{pdf_name}.pdf"
                response = s3_client.head_object(Bucket=bucket_name, Key=pdf_key)
                
                pdf_details.append({
                    "name": pdf_name,
                    "size": response.get('ContentLength', 0),
                    "lastModified": response.get('LastModified', '').isoformat(),
                    "processed": True
                })
                print(f"Added details for PDF: {pdf_name}")  # Debug log
            except ClientError:
                print(f"Could not get details for PDF: {pdf_name}")  # Debug log
                continue
        
        print(f"Returning {len(pdf_details)} processed PDFs")  # Debug log
        return JSONResponse(
            content={"pdfs": pdf_details},
            status_code=200
        )
        
    except Exception as e:
        error_msg = f"Failed to list processed PDFs: {str(e)}"
        print(f"Error: {error_msg}")  # Debug log
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@router.delete("/delete/{full_path:path}")
async def delete_file(full_path: str):
    """Delete a PDF file and its associated files from S3"""
    try:
        print(f"Attempting to delete: {full_path}")
        # Extract the base name from the full path
        filename = full_path.split('/')[-1]
        base_name = Path(filename).stem
        print(f"Base name: {base_name}")
        
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        
        # List all objects under this base name
        prefix = f"documents/{base_name}/"
        print(f"Listing objects with prefix: {prefix}")
        
        try:
            objects_to_delete = []
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        print(f"Found object to delete: {obj['Key']}")
                        objects_to_delete.append({'Key': obj['Key']})
            
            if objects_to_delete:
                print(f"Deleting {len(objects_to_delete)} objects")
                s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={
                        'Objects': objects_to_delete,
                        'Quiet': True
                    }
                )
                print("Successfully deleted all objects")
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": f"Successfully deleted {filename} and all associated files",
                        "deleted_count": len(objects_to_delete)
                    }
                )
            else:
                print("No objects found to delete")
                raise HTTPException(
                    status_code=404,
                    detail=f"No files found for {filename}"
                )
                
        except ClientError as e:
            print(f"S3 error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete files: {str(e)}"
            )
            
    except Exception as e:
        print(f"Error in delete_file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )
