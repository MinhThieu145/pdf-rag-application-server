import sys
import os
from pathlib import Path

# Add the server directory to Python path
server_dir = str(Path(__file__).parent)
if server_dir not in sys.path:
    sys.path.append(server_dir)

from fastapi import FastAPI, Request, APIRouter, HTTPException, Query, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from controllers import course_controller, property_graph_controller, pdf_chatting_controller
from controllers.text_splitting_controller import router as text_split_router
from controllers.text_embedding_controller import router as text_embed_router
from controllers.text_processing_controller import router as text_process_router
from controllers.essay_controller import router as essay_router
from controllers.evidence_extraction_controller import router as evidence_router
from controllers.s3_controller import router as s3_router
from controllers.pdf_controller import router as pdf_router
from tests.routes.evidence_test_routes import router as evidence_test_router
from schemas.course_schema import CourseCreate, CourseUpdate, VideoUploadRequest
import boto3
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize DynamoDB tables
# @app.on_event("startup")
# async def startup_event():
#     create_tables()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware for handling large files
@app.middleware("http")
async def add_custom_headers(request, call_next):
    response = await call_next(request)
    response.headers["Large-Allocation"] = "true"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# Configure server settings
app.state.max_header_size = 32768  # 32KB for headers
app.state.max_request_size = 100 * 1024 * 1024  # 100MB max request size

# Include routes
app.include_router(course_controller.router, prefix="/api/courses", tags=["courses"])
app.include_router(pdf_router, prefix="/api/pdf", tags=["PDF"])
app.include_router(property_graph_controller.router, prefix="/api/graph", tags=["graph"])
app.include_router(pdf_chatting_controller.router, prefix="/api/chat", tags=["chat"])
app.include_router(text_split_router, prefix="/api/split", tags=["text-splitting"])
app.include_router(text_embed_router, prefix="/api/embed", tags=["text-embedding"])
app.include_router(text_process_router, prefix="/api/process", tags=["text-processing"])
app.include_router(essay_router, prefix="/api/essay", tags=["Essay"])
app.include_router(evidence_router, prefix="/api/evidence", tags=["Evidence"])
app.include_router(s3_router, prefix="/api/s3", tags=["S3"])
app.include_router(evidence_test_router, prefix="/api/test", tags=["Tests"])

# Test routes for development and debugging
class S3TestRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/api/s3/test", tags=["S3 Testing"])
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.bucket_name = os.getenv('AWS_BUCKET_NAME')
        
        # Register routes
        self.router.add_api_route("/list", self.list_s3_contents, methods=["GET"])
        self.router.add_api_route("/check/{filepath:path}", self.check_s3_file, methods=["GET"])
        self.router.add_api_route("/download/{filepath:path}", self.download_s3_file, methods=["GET"])

    async def list_s3_contents(self, prefix: str = Query("evidence/", description="Directory prefix to list")):
        """List contents of S3 bucket with given prefix"""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return {"files": []}
                
            files = [obj['Key'] for obj in response['Contents']]
            return {"files": files}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list S3 contents: {str(e)}"
            )

    async def check_s3_file(self, filepath: str):
        """Check if a file exists in S3 and get its metadata"""
        try:
            response = self.s3.head_object(
                Bucket=self.bucket_name,
                Key=filepath
            )
            
            return {
                "exists": True,
                "size": response.get('ContentLength', 0),
                "last_modified": response.get('LastModified', None),
                "metadata": response.get('Metadata', {})
            }
        except Exception as e:
            if hasattr(e, 'response') and e.response['Error']['Code'] == '404':
                return {"exists": False}
            raise HTTPException(
                status_code=500,
                detail=f"Failed to check file: {str(e)}"
            )

    async def download_s3_file(self, filepath: str):
        """Download a file from S3 and return its contents"""
        try:
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=filepath
            )
            
            return {
                "content_type": response.get('ContentType', 'application/octet-stream'),
                "content_length": response.get('ContentLength', 0),
                "last_modified": response.get('LastModified', None),
                "metadata": response.get('Metadata', {})
            }
        except Exception as e:
            if hasattr(e, 'response') and e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {filepath}"
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download file: {str(e)}"
            )

# Initialize test router
s3_test_router = S3TestRouter()

# Include routes
app.include_router(course_controller.router, prefix="/api/courses", tags=["courses"])
app.include_router(pdf_router, prefix="/api/pdf", tags=["PDF"])
app.include_router(property_graph_controller.router, prefix="/api/graph", tags=["graph"])
app.include_router(pdf_chatting_controller.router, prefix="/api/chat", tags=["chat"])
app.include_router(text_split_router, prefix="/api/split", tags=["text-splitting"])
app.include_router(text_embed_router, prefix="/api/embed", tags=["text-embedding"])
app.include_router(text_process_router, prefix="/api/process", tags=["text-processing"])
app.include_router(essay_router, prefix="/api/essay", tags=["Essay"])
app.include_router(evidence_router, prefix="/api/evidence", tags=["Evidence"])
app.include_router(s3_router, prefix="/api/s3", tags=["S3"])
app.include_router(evidence_test_router, prefix="/api/test", tags=["Tests"])
app.include_router(s3_test_router.router)