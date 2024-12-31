from typing import List, Optional
from pydantic import BaseModel

class CourseCreate(BaseModel):
    teacherId: str
    teacherName: str

class CourseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    image: Optional[str] = None
    price: Optional[float] = None
    level: Optional[str] = None
    status: Optional[str] = None
    sections: Optional[List[dict]] = None

class VideoUploadRequest(BaseModel):
    fileName: str
    fileType: str

# Response Models
class CourseResponse(BaseModel):
    message: str
    data: dict

class CoursesResponse(BaseModel):
    message: str
    data: List[dict]
