from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from pydantic import BaseModel
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Pydantic models
class ParagraphBase(BaseModel):
    text: str
    name: Optional[str] = None
    purposes: Optional[List[str]] = []
    evidence_source: Optional[str] = None

class ParagraphCreate(ParagraphBase):
    pass

class Paragraph(ParagraphBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Temporary storage (replace with database in production)
paragraphs_db = [
    {
        "id": 1,
        "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "name": "Introduction",
        "purposes": ["Introduce the topic", "Set context"],
        "evidence_source": "None",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    },
    {
        "id": 2,
        "text": "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "name": "Body Paragraph 1",
        "purposes": ["Present main argument"],
        "evidence_source": "Research paper",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    },
    {
        "id": 3,
        "text": "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "name": "Conclusion",
        "purposes": ["Summarize points", "Conclude argument"],
        "evidence_source": "None",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
]

@router.get("/paragraphs/", response_model=List[Paragraph])
async def get_paragraphs():
    """Get all paragraphs"""
    try:
        return paragraphs_db
    except Exception as e:
        logger.error(f"Error getting paragraphs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/paragraphs/{paragraph_id}", response_model=Paragraph)
async def get_paragraph(paragraph_id: int):
    """Get a specific paragraph by ID"""
    try:
        paragraph = next((p for p in paragraphs_db if p["id"] == paragraph_id), None)
        if paragraph is None:
            raise HTTPException(status_code=404, detail="Paragraph not found")
        return paragraph
    except Exception as e:
        logger.error(f"Error getting paragraph {paragraph_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/paragraphs/", response_model=Paragraph)
async def create_paragraph(paragraph: ParagraphCreate):
    """Create a new paragraph"""
    try:
        new_id = max(p["id"] for p in paragraphs_db) + 1 if paragraphs_db else 1
        new_paragraph = {
            "id": new_id,
            **paragraph.dict(),
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        paragraphs_db.append(new_paragraph)
        return new_paragraph
    except Exception as e:
        logger.error(f"Error creating paragraph: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/paragraphs/{paragraph_id}", response_model=Paragraph)
async def update_paragraph(paragraph_id: int, paragraph: ParagraphCreate):
    """Update a paragraph"""
    try:
        for i, p in enumerate(paragraphs_db):
            if p["id"] == paragraph_id:
                paragraphs_db[i] = {
                    **p,
                    **paragraph.dict(),
                    "updated_at": datetime.now()
                }
                return paragraphs_db[i]
        raise HTTPException(status_code=404, detail="Paragraph not found")
    except Exception as e:
        logger.error(f"Error updating paragraph {paragraph_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/paragraphs/{paragraph_id}")
async def delete_paragraph(paragraph_id: int):
    """Delete a paragraph"""
    try:
        for i, p in enumerate(paragraphs_db):
            if p["id"] == paragraph_id:
                del paragraphs_db[i]
                return JSONResponse(content={"message": "Paragraph deleted successfully"})
        raise HTTPException(status_code=404, detail="Paragraph not found")
    except Exception as e:
        logger.error(f"Error deleting paragraph {paragraph_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/paragraphs/reorder", response_model=List[Paragraph])
async def reorder_paragraphs(paragraph_ids: List[int]):
    """Reorder paragraphs based on the provided list of IDs"""
    try:
        # Verify all IDs exist
        if not all(any(p["id"] == pid for p in paragraphs_db) for pid in paragraph_ids):
            raise HTTPException(status_code=404, detail="One or more paragraphs not found")
        
        # Create a new ordered list
        reordered = []
        for pid in paragraph_ids:
            paragraph = next(p for p in paragraphs_db if p["id"] == pid)
            reordered.append(paragraph)
        
        # Update the database
        paragraphs_db.clear()
        paragraphs_db.extend(reordered)
        
        return reordered
    except Exception as e:
        logger.error(f"Error reordering paragraphs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
