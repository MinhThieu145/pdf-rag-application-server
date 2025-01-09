import s3fs
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse

import os

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


# AWS S3 configuration
bucket_name = "pdf-chat-application"
s3_prefix = "documents/Santiago_1/"  # Folder in S3

# Create an S3FileSystem instance
fs = s3fs.S3FileSystem()

# List files in the S3 folder
files = fs.ls(f"s3://{bucket_name}/{s3_prefix}")
print("Files found in S3:", files)

# Download files locally for processing by SimpleDirectoryReader
local_folder = "./temp_s3_documents"

# empty the folder
for filename in os.listdir(local_folder):
    file_path = os.path.join(local_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f"Failed to delete {file_path}. Reason: {str(e)}")

for file_path in files:
    file_name = file_path.split("/")[-1]
    fs.get(file_path, f"{local_folder}/{file_name}")

# print the llama cloud api key
print(os.getenv("LLAMA_CLOUD_API_KEY"))

# llama parser
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="text",
    verbose=True,

)

file_extractor = {'.pdf' : parser}

# Load documents using SimpleDirectoryReader
documents = SimpleDirectoryReader(local_folder, file_extractor=file_extractor).load_data()

print()
print("loaded JSON")
print(documents)
print()
print('--------------------------------------------------------------------')
# print the text
for i in range(len(documents)):
    print(f"Document {i}:")
    print(documents[i].text)
