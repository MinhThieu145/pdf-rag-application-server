from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers import course_controller, pdf_controller, property_graph_controller, pdf_chatting_controller
from controllers.text_splitting_controller import router as text_split_router
from controllers.text_embedding_controller import router as text_embed_router
from controllers.text_processing_controller import router as text_process_router
from schemas.course_schema import CourseCreate, CourseUpdate, VideoUploadRequest
# from config.db_config import create_tables

# Initialize FastAPI app
app = FastAPI()

# Initialize DynamoDB tables
# @app.on_event("startup")
# async def startup_event():
#     create_tables()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using "*" for origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routes
# app.include_router(course_controller.router, prefix="/api/courses", tags=["Courses"])
app.include_router(pdf_controller.router, prefix="/api/pdf", tags=["PDF"])
app.include_router(property_graph_controller.router, prefix="/api/property-graph", tags=["Property Graph"])
app.include_router(pdf_chatting_controller.router, prefix="/api/pdf-chat", tags=["PDF Chat"])
app.include_router(text_split_router, prefix="/api/text-split", tags=["text-split"])
app.include_router(text_embed_router, prefix="/api/embed", tags=["embeddings"])
app.include_router(text_process_router, prefix="/api/process", tags=["text-processing"])