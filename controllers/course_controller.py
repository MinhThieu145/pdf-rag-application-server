from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
import boto3
from decimal import Decimal
import uuid
from datetime import datetime
from config.db_config import get_dynamodb_resource
from schemas.course_schema import CourseCreate, CourseUpdate, VideoUploadRequest, CourseResponse, CoursesResponse

router = APIRouter()
dynamodb = get_dynamodb_resource()
table = dynamodb.Table('Courses')
s3 = boto3.client('s3')

@router.post("/courses/", response_model=CourseResponse)
async def create_course(course: CourseCreate):
    course_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    course_item = {
        'courseId': course_id,
        'teacherId': course.teacherId,
        'teacherName': course.teacherName,
        'title': "Untitled Course",
        'description': "",
        'category': "Uncategorized",
        'image': "",
        'price': str(Decimal(0)),  
        'level': "Beginner",
        'status': "Draft",
        'sections': [],
        'enrollments': [],
        'createdAt': timestamp,
        'updatedAt': timestamp
    }
    
    try:
        table.put_item(Item=course_item)
        return CourseResponse(**course_item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/courses/{course_id}", response_model=CourseResponse)
async def get_course(course_id: str):
    try:
        response = table.get_item(Key={'courseId': course_id})
        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="Course not found")
        return CourseResponse(**response['Item'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/courses/", response_model=CoursesResponse)
async def list_courses(category: Optional[str] = None):
    try:
        if category and category != "all":
            response = table.query(
                IndexName='CategoryIndex',
                KeyConditionExpression='category = :category',
                ExpressionAttributeValues={':category': category}
            )
        else:
            response = table.scan()
        courses = response.get('Items', [])
        return {
            "message": "Courses retrieved successfully",
            "data": courses
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/courses/{course_id}", response_model=CourseResponse)
async def update_course(course_id: str, course_update: CourseUpdate):
    timestamp = datetime.utcnow().isoformat()
    update_expression = "SET "
    expression_attribute_values = {}
    expression_attribute_names = {}
    
    update_attrs = {
        '#title': 'title',
        '#description': 'description',
        '#category': 'category',
        '#price': 'price',
        '#level': 'level',
        '#status': 'status',
        '#updatedAt': 'updatedAt'
    }
    
    for attr_name, attr_value in {
        'title': course_update.title,
        'description': course_update.description,
        'category': course_update.category,
        'price': str(Decimal(course_update.price)),
        'level': course_update.level,
        'status': course_update.status,
        'updatedAt': timestamp
    }.items():
        if attr_value is not None:
            expression_attribute_names[f'#{attr_name}'] = attr_name
            expression_attribute_values[f':{attr_name}'] = attr_value
            update_expression += f'#{attr_name} = :{attr_name}, '
    
    update_expression = update_expression.rstrip(', ')
    
    try:
        response = table.update_item(
            Key={'courseId': course_id},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ExpressionAttributeNames=expression_attribute_names,
            ReturnValues='ALL_NEW'
        )
        return CourseResponse(**response['Attributes'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/courses/{course_id}")
async def delete_course(course_id: str):
    try:
        table.delete_item(Key={'courseId': course_id})
        return {"message": "Course deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/courses/{course_id}/upload-video")
async def get_upload_video_url(course_id: str, request: VideoUploadRequest):
    try:
        if not os.getenv("S3_BUCKET_NAME"):
            raise HTTPException(status_code=500, detail="S3 bucket name not configured")

        unique_id = str(uuid.uuid4())
        s3_key = f"videos/{unique_id}/{request.fileName}"

        presigned_url = s3.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': os.getenv("S3_BUCKET_NAME"),
                'Key': s3_key,
                'ContentType': request.fileType
            },
            ExpiresIn=60
        )

        video_url = f"{os.getenv('CLOUDFRONT_DOMAIN')}/videos/{unique_id}/{request.fileName}"

        return {
            "message": "Upload URL generated successfully",
            "data": {
                "uploadUrl": presigned_url,
                "videoUrl": video_url
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
