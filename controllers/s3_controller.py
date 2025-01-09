from fastapi import APIRouter, HTTPException
import os
import boto3
from botocore.exceptions import ClientError
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize router and S3 client
router = APIRouter()
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

@router.get("/list")
async def list_files():
    """List all files in S3 under evidence/ directory"""
    try:
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='evidence/'
        )
        
        if 'Contents' not in response:
            return {"files": []}
            
        files = [obj['Key'] for obj in response['Contents']]
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {str(e)}"
        )

@router.get("/check/{filepath}")
async def check_file(filepath: str):
    """Check if a file exists in S3 and return its metadata"""
    try:
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        response = s3_client.head_object(
            Bucket=bucket_name,
            Key=filepath
        )
        
        return {
            "exists": True,
            "size": response.get('ContentLength', 0),
            "last_modified": response.get('LastModified', None),
            "metadata": response.get('Metadata', {})
        }
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return {"exists": False}
        raise HTTPException(
            status_code=500,
            detail=f"Error checking file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error checking file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check file: {str(e)}"
        )

@router.get("/download/{filepath}")
async def download_file(filepath: str):
    """Download a file from S3"""
    try:
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=filepath
        )
        
        return {
            "content_type": response.get('ContentType', 'application/octet-stream'),
            "content_length": response.get('ContentLength', 0),
            "last_modified": response.get('LastModified', None),
            "metadata": response.get('Metadata', {})
        }
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {filepath}"
            )
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download file: {str(e)}"
        )
