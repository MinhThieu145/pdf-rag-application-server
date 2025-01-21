from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import openai
import os
from dotenv import load_dotenv
import asyncio
from typing import Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from the server/.env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
logger.info(f"Loading environment variables from {env_path}")

# Configure OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found in environment variables")

assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
if not assistant_id:
    logger.error(f"OPENAI_ASSISTANT_ID not found in environment variables. Env file path: {env_path}")
    raise ValueError("OPENAI_ASSISTANT_ID not found in environment variables")

client = openai.AsyncOpenAI(api_key=api_key)

router = APIRouter(
    tags=["Chat"],
    responses={404: {"description": "Not found"}},
)

class ChatMessage(BaseModel):
    message: str
    threadId: str

class ChatResponse(BaseModel):
    message: str

class ThreadResponse(BaseModel):
    threadId: str

class ErrorResponse(BaseModel):
    detail: str

async def wait_on_run(thread_id: str, run_id: str) -> Dict[str, Any]:
    """Wait for a run to complete and handle timeouts."""
    max_attempts = 60  # Maximum number of attempts (60 seconds)
    attempt = 0
    
    while attempt < max_attempts:
        try:
            run_status = await client.beta.threads.runs.retrieve(
                run_id=run_id,
                thread_id=thread_id
            )
            if run_status.status in ["completed", "failed", "expired", "cancelled"]:
                return run_status
            elif run_status.status in ["queued", "in_progress"]:
                await asyncio.sleep(1)
                attempt += 1
            else:
                error_msg = f"Unexpected run status: {run_status.status}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_msg
                )
        except Exception as e:
            error_msg = f"Error checking run status: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
    
    error_msg = "Request timed out waiting for assistant response"
    logger.error(error_msg)
    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=error_msg
    )

@router.post("/chat", response_model=ChatResponse, responses={
    500: {"model": ErrorResponse},
    504: {"model": ErrorResponse}
})
async def process_chat_message(chat_message: ChatMessage):
    """Process a chat message and return the assistant's response."""
    try:
        logger.info(f"Processing chat message for thread {chat_message.threadId}")
        
        # Add the message to the thread
        await client.beta.threads.messages.create(
            thread_id=chat_message.threadId,
            role="user",
            content=chat_message.message
        )

        # Create a run with the assistant
        assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
        if not assistant_id:
            error_msg = "OPENAI_ASSISTANT_ID not found in environment variables"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

        run = await client.beta.threads.runs.create(
            thread_id=chat_message.threadId,
            assistant_id=assistant_id
        )

        # Wait for the run to complete
        run_status = await wait_on_run(chat_message.threadId, run.id)
        
        if run_status.status != "completed":
            error_msg = f"Run failed with status: {run_status.status}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

        # Get the messages after the run completes
        messages = await client.beta.threads.messages.list(chat_message.threadId)
        if not messages.data:
            error_msg = "No response received from assistant"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

        last_message = messages.data[0]
        if not last_message.content or not last_message.content[0].text:
            error_msg = "Invalid response format from assistant"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

        logger.info("Successfully processed chat message")
        return ChatResponse(message=last_message.content[0].text.value)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to process chat message: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

@router.post("/create-thread", response_model=ThreadResponse, responses={
    500: {"model": ErrorResponse}
})
async def create_thread():
    """Create a new thread for the chat conversation."""
    try:
        logger.info("Creating new chat thread")
        thread = await client.beta.threads.create()
        logger.info(f"Successfully created thread {thread.id}")
        return ThreadResponse(threadId=thread.id)
    except Exception as e:
        error_msg = f"Failed to create thread: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

@router.delete("/thread/{thread_id}", responses={
    500: {"model": ErrorResponse}
})
async def delete_thread(thread_id: str):
    """Delete a thread."""
    try:
        logger.info(f"Deleting thread {thread_id}")
        await client.beta.threads.delete(thread_id=thread_id)
        logger.info(f"Successfully deleted thread {thread_id}")
        return {"status": "success"}
    except Exception as e:
        error_msg = f"Failed to delete thread: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )
