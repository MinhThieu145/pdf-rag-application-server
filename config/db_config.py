import boto3
import os

# DynamoDB Configuration
def get_dynamodb_client():
    """Get DynamoDB client"""
    return boto3.client(
        'dynamodb',
        endpoint_url='http://localhost:8000',  # Local DynamoDB endpoint
        region_name='local',
        aws_access_key_id='dummy',
        aws_secret_access_key='dummy'
    )

def get_dynamodb_resource():
    """Get DynamoDB resource"""
    return boto3.resource(
        'dynamodb',
        endpoint_url='http://localhost:8000',  # Local DynamoDB endpoint
        region_name='local',
        aws_access_key_id='dummy',
        aws_secret_access_key='dummy'
    )

def create_tables():
    """Create DynamoDB tables if they don't exist"""
    dynamodb = get_dynamodb_resource()
    
    # Courses table
    try:
        table = dynamodb.create_table(
            TableName='Courses',
            KeySchema=[
                {'AttributeName': 'courseId', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'courseId', 'AttributeType': 'S'},
                {'AttributeName': 'category', 'AttributeType': 'S'}
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'CategoryIndex',
                    'KeySchema': [
                        {'AttributeName': 'category', 'KeyType': 'HASH'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'},
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        print("Creating Courses table...")
        table.wait_until_exists()
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print("Courses table already exists")

    # Transactions table
    try:
        table = dynamodb.create_table(
            TableName='Transactions',
            KeySchema=[
                {'AttributeName': 'transactionId', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'transactionId', 'AttributeType': 'S'}
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        print("Creating Transactions table...")
        table.wait_until_exists()
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print("Transactions table already exists")

    # UserCourseProgress table
    try:
        table = dynamodb.create_table(
            TableName='UserCourseProgress',
            KeySchema=[
                {'AttributeName': 'userId', 'KeyType': 'HASH'},
                {'AttributeName': 'courseId', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'userId', 'AttributeType': 'S'},
                {'AttributeName': 'courseId', 'AttributeType': 'S'}
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        print("Creating UserCourseProgress table...")
        table.wait_until_exists()
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print("UserCourseProgress table already exists")
