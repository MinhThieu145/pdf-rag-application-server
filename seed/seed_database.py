import os
import sys
import json
from typing import Any, Dict, List, Union
from decimal import Decimal, InvalidOperation
import boto3

# Dynamically add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom database configuration
from config.db_config import get_dynamodb_resource


def convert_to_dynamodb_type(value: Any) -> Any:
    """
    Convert various data types to DynamoDB-compatible types.
    """
    if isinstance(value, bool):
        return value  # Handle booleans first to prevent conversion to Decimal
    elif isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            print(f"Failed to convert numeric value: {value}")
            raise
    elif isinstance(value, dict):
        return {k: convert_to_dynamodb_type(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_dynamodb_type(item) for item in value]
    return value


def validate_data(data: List[Dict[str, Any]], schema: Dict[str, Any]) -> bool:
    """
    Basic validation for the data structure against a schema.
    """
    for record in data:
        for field, field_schema in schema.items():
            if field not in record:
                if field_schema.get('required', True):
                    print(f"Missing required field '{field}' in record: {record}")
                    return False
            else:
                value = record[field]
                expected_type = field_schema['type']
                if not isinstance(value, expected_type):
                    if isinstance(expected_type, tuple):
                        if not any(isinstance(value, t) for t in expected_type):
                            print(f"Field '{field}' has incorrect type in record: {record}")
                            return False
                    else:
                        print(f"Field '{field}' has incorrect type in record: {record}")
                        return False
    return True


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file located in the 'data' directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, "data", file_path)

    try:
        with open(full_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {full_path}")
        raise
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {full_path}")
        raise


def seed_table(table_name: str, data_file: str, schema: Dict[str, Dict[str, Any]]):
    """
    Seed data into the specified DynamoDB table.
    """
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(table_name)

    data = load_json_data(data_file)

    # Validate the data before seeding
    if not validate_data(data, schema):
        print(f"Data validation failed for file: {data_file}")
        return

    for record in data:
        try:
            # Convert data to DynamoDB-compatible types
            converted_record = convert_to_dynamodb_type(record)
            table.put_item(Item=converted_record)
        except Exception as e:
            print(f"Failed to seed record: {record}. Error: {str(e)}")
            raise
    print(f"Successfully seeded data into {table_name}")


def seed_courses():
    """
    Seed course data into the 'Courses' table.
    """
    print("Seeding courses data...")
    schema = {
        "courseId": {"type": str, "required": True},
        "teacherId": {"type": str, "required": True},
        "teacherName": {"type": str, "required": True},
        "title": {"type": str, "required": True},
        "description": {"type": str, "required": True},
        "category": {"type": str, "required": True},
        "image": {"type": str, "required": True},
        "price": {"type": (int, float), "required": True},
        "level": {"type": str, "required": True},
        "status": {"type": str, "required": True},
        "enrollments": {"type": list, "required": True},
        "sections": {"type": list, "required": True}
    }
    seed_table("Courses", "courses.json", schema)


def seed_transactions():
    """
    Seed transaction data into the 'Transactions' table.
    """
    print("Seeding transactions data...")
    schema = {
        "transactionId": {"type": str, "required": True},
        "userId": {"type": str, "required": True},
        "courseId": {"type": str, "required": True},
        "amount": {"type": (int, float), "required": True}
    }
    seed_table("Transactions", "transactions.json", schema)


def seed_user_progress():
    """
    Seed user course progress data into the 'UserCourseProgress' table.
    """
    print("Seeding user course progress data...")
    schema = {
        "userId": {"type": str, "required": True},
        "courseId": {"type": str, "required": True},
        "enrollmentDate": {"type": str, "required": True},
        "overallProgress": {"type": (int, float), "required": True},
        "progress": {"type": (int, float), "required": True},
        "sections": {"type": list, "required": True},
        "lastAccessedTimestamp": {"type": str, "required": True}
    }
    seed_table("UserCourseProgress", "userCourseProgress.json", schema)


def seed_all():
    """
    Seed all data into the database.
    """
    print("Starting database seeding...")
    try:
        seed_courses()
        seed_transactions()
        seed_user_progress()
        print("Database seeding completed successfully!")
    except Exception as e:
        print(f"Database seeding failed: {e}")
        raise


if __name__ == "__main__":
    seed_all()
