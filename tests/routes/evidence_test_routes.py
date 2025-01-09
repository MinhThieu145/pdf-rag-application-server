import sys
import os
from pathlib import Path

import boto3



# Add the server directory to Python path
server_dir = str(Path(__file__).parent.parent.parent)
if server_dir not in sys.path:
    sys.path.append(server_dir)

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(tags=["Evidence Tests"])

@router.get("/status")
async def test_status() -> Dict[str, str]:
    """Test endpoint to check if the evidence routes are working"""
    return {"status": "Evidence test routes are operational"}


# route for parsing const evidenceResponse = await api.get<JsonData>(`/parse/${fileInfo.file.name}`)

### AWS CONFIGURATION ###

# load the bucket name from the environment
bucket_name = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)


### TESTING ROUTES ###

# this is for parsing file from pdf to text
@router.get("/parse/{filename}")
async def parse_pdf(filename: str) -> Dict[str, Any]:
    """Test endpoint to parse a PDF file"""
    try:
        # Construct the S3 key
        base_name = Path(filename).stem
        pdf_key = f"documents/{base_name}/{filename}"
        
        # Check if file exists in S3
        try:
            s3.head_object(Bucket=bucket_name, Key=pdf_key)
        except s3.exceptions.ClientError as e:
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
        return {
            "message": f"PDF file {filename} found in S3",
            "location": f"s3://{bucket_name}/{pdf_key}",
            "exists": True
        }

    except Exception as e:
        logger.error(f"Error in parse_pdf: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )    


# function for parsing file
def parse_file(s3_prefix):

    # load file from s3 to SimpleDirectoryReader
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    import s3fs
    from llama_index.core import SimpleDirectoryReader
    import os

    # Create an S3FileSystem instance
    fs = s3fs.S3FileSystem()

    files = fs.ls(f"s3://{bucket_name}/{s3_prefix}")

    # Load documents using SimpleDirectoryReader
    documents = SimpleDirectoryReader(local_folder).load_data()

    # Use the documents in your LlamaIndex pipeline
    print("Loaded documents:", documents)



