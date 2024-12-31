from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
import os
from openai import OpenAI
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel
import time
from helpers.pinecone_helper import search_vectors
from helpers.embedding_helper import create_embedding
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Store thread IDs for different users (in a real application, this should be in a database)
THREADS_FILE = "pdf_chat_threads.json"

# Pydantic models for request validation
class UserIdRequest(BaseModel):
    user_id: str

class ChatRequest(BaseModel):
    user_id: str
    assistant_id: str
    message: str
    pdf_name: Optional[str] = None

def load_threads() -> Dict[str, str]:
    try:
        if os.path.exists(THREADS_FILE):
            with open(THREADS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading threads: {str(e)}")
        return {}

def save_threads(threads: Dict[str, str]):
    try:
        with open(THREADS_FILE, 'w') as f:
            json.dump(threads, f)
    except Exception as e:
        logger.error(f"Error saving threads: {str(e)}")

# Initialize threads from file
user_threads = load_threads()

@router.post("/create-assistant")
async def create_assistant():
    try:
        assistant = client.beta.assistants.create(
            name="PDF Expert",
            instructions="You are an expert at analyzing PDF documents. Help users understand the content, extract information, and answer questions about the documents.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-3.5-turbo"
        )
        return JSONResponse(content={"assistant_id": assistant.id}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-thread")
async def create_thread(request: UserIdRequest):
    try:
        # Reload threads from file to get latest state
        global user_threads
        user_threads = load_threads()
        
        thread = client.beta.threads.create()
        user_threads[request.user_id] = thread.id
        save_threads(user_threads)
        return JSONResponse(content={"thread_id": thread.id}, status_code=200)
    except Exception as e:
        print(f"Error in create_thread: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-thread/{user_id}")
async def get_thread(user_id: str):
    try:
        # Reload threads from file to get latest state
        global user_threads
        user_threads = load_threads()
        
        thread_id = user_threads.get(user_id)
        if thread_id:
            try:
                # Verify thread still exists in OpenAI
                thread = client.beta.threads.retrieve(thread_id)
                
                # Get all messages in the thread
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                chat_history = []
                
                for msg in messages.data:
                    if msg.content:
                        chat_history.append({
                            "role": msg.role,
                            "content": msg.content[0].text.value
                        })
                
                return JSONResponse(
                    content={
                        "thread_id": thread_id,
                        "chat_history": chat_history[::-1]  # Reverse to get chronological order
                    }, 
                    status_code=200
                )
            except Exception:
                # Thread doesn't exist in OpenAI anymore, remove it
                del user_threads[user_id]
                save_threads(user_threads)
                return JSONResponse(content={"thread_id": None}, status_code=404)
        return JSONResponse(content={"thread_id": None}, status_code=404)
    except Exception as e:
        print(f"Error in get_thread: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that uses selected PDF context for responses
    """
    try:
        # Create embedding for the user's question
        query_embedding = create_embedding(request.message)
        
        # Search Pinecone with PDF filter if specified
        search_filter = {"pdf_name": request.pdf_name} if request.pdf_name else None
        relevant_chunks = await search_vectors(
            query_vector=query_embedding,
            filter=search_filter,
            top_k=5
        )
        
        # Format context from relevant chunks
        context = "\n\n".join([chunk.metadata["text"] for chunk in relevant_chunks])
        
        # Get thread ID for user
        thread_id = user_threads.get(request.user_id)
        if not thread_id:
            return JSONResponse(
                content={"error": "Thread not found"}, 
                status_code=404
            )

        # Add user message with context
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=f"""Use the following context to answer my question. If the answer cannot be found in the context, say so.
            
            Context:
            {context}
            
            Question: {request.message}"""
        )

        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=request.assistant_id
        )

        # Wait for the run to complete
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            elif run_status.status == 'failed':
                raise Exception("Assistant run failed")
            time.sleep(1)

        # Get the latest message
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        latest_message = messages.data[0]  # Most recent message

        return JSONResponse(
            content={
                "response": latest_message.content[0].text.value,
                "context_used": bool(relevant_chunks)
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat: {str(e)}"
        )

@router.delete("/thread/{thread_id}")
async def delete_thread(thread_id: str):
    try:
        client.beta.threads.delete(thread_id)
        # Remove from user_threads if exists
        for user_id, tid in list(user_threads.items()):
            if tid == thread_id:
                del user_threads[user_id]
        save_threads(user_threads)  # Save to file
        return JSONResponse(content={"status": "success"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
