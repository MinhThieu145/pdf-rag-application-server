from typing import List, Optional
from dynamoose import model, Model, Schema

class ChapterProgress(Model):
    class Meta:
        table_name = "chapter_progress"
        timestamps = True

    chapterId: str
    completed: bool

class SectionProgress(Model):
    class Meta:
        table_name = "section_progress"
        timestamps = True

    sectionId: str
    chapters: List[ChapterProgress]

class UserCourseProgress(Model):
    class Meta:
        table_name = "user_course_progress"
        timestamps = True

    userId: str
    courseId: str
    enrollmentDate: str
    overallProgress: float
    sections: List[SectionProgress]
    lastAccessedTimestamp: str
