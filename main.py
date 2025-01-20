import sys
import os
from pathlib import Path

# Add the server directory to Python path
server_dir = str(Path(__file__).parent)
if server_dir not in sys.path:
    sys.path.append(server_dir)

from fastapi import FastAPI, Request, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from controllers.essay_generation_controller import router as essay_generation_router
from controllers.evidence_extraction_controller import router as evidence_router
from controllers.s3_controller import router as s3_router
import boto3
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

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

# Include only the used routes
app.include_router(evidence_router, prefix="/api/evidence", tags=["Evidence"])
app.include_router(s3_router, prefix="/api/s3", tags=["S3"])
app.include_router(essay_generation_router, prefix="/api/essay-generation", tags=["Essay Generation"])