from typing import List, Optional
from dynamoose import model, Model, Schema

class Comment(Model):
    class Meta:
        table_name = "comments"
        timestamps = True

    commentId: str
    userId: str
    text: str
    timestamp: str

class Chapter(Model):
    class Meta:
        table_name = "chapters"
        timestamps = True

    chapterId: str
    type: str  # "Text" | "Quiz" | "Video"
    title: str
    content: str
    comments: Optional[List[Comment]]
    video: Optional[str]

class Section(Model):
    class Meta:
        table_name = "sections"
        timestamps = True

    sectionId: str
    sectionTitle: str
    sectionDescription: Optional[str]
    chapters: List[Chapter]

class Course(Model):
    class Meta:
        table_name = "courses"
        timestamps = True

    courseId: str
    teacherId: str
    teacherName: str
    title: str
    description: Optional[str]
    category: str
    image: Optional[str]
    price: Optional[float]
    level: str
    sections: List[Section]
    enrollments: List[dict]
    status: str
