"""
Essay Generation Controller

This module handles the generation of argumentative essays using GPT-4.
It provides endpoints for generating well-structured essays based on provided context and topics.

Key Features:
- Essay structure generation with introduction, body paragraphs, and conclusion
- Evidence-based argumentation using provided context
- Detailed planning and organization
- Word count control

Environment Variables Required:
- OPENAI_API_KEY: OpenAI API key
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List, Optional
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI router
router = APIRouter(tags=["Essay Generation"])

# Define Pydantic models
class DetailedEssayResponse(BaseModel):
    essay_content: str

class EssayGenerationRequest(BaseModel):
    context: str
    topic: str
    word_count: int

@router.get("/test")
async def test_endpoint():
    return {"message": "Essay generation router is working"}

@router.post("/generate", response_model=DetailedEssayResponse)
async def generate_essay(request: EssayGenerationRequest) -> DetailedEssayResponse:
    """
    Generate a detailed argumentative essay based on provided context and topic.
    """
    try:
        print("\n=== Essay Generation Request ===")
        print(f"Topic: {request.topic}")
        print(f"Word Count: {request.word_count}")
        print("\nContext:")
        print("----------------------------------------")
        print(request.context)
        print("----------------------------------------\n")
        
        # Ensure OpenAI API key is set
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        client = OpenAI(api_key=openai_api_key)

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert essay writer specializing in creating well-structured, compelling argumentative essays. Generate a complete essay based on the provided evidence and topic, adhering to academic writing standards and the specified word count."
                },
                {
                    "role": "user", 
                    "content": f"Evidence with Reasoning:\n{request.context}\n\nTopic: {request.topic}\n\nWord Count: {request.word_count}"
                }
            ],
            temperature=0.5,
            max_tokens=request.word_count * 3
        )

        print("Received response from GPT-4")
        content = response.choices[0].message.content
        print("Response content:", content)

        return DetailedEssayResponse(essay_content=content)

    except Exception as e:
        print(f"Error in essay generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate essay: {str(e)}")
