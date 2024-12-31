from fastapi import APIRouter
import json
import os
import shutil
from llama_index.core import SimpleDirectoryReader
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

router = APIRouter()

# Initialize models
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4")

def ensure_temp_directory(directory: str = "temp_text_for_property") -> str:
    """
    Ensure the temporary directory exists and is empty
    """
    # Get absolute path
    base_path = os.path.join(os.getcwd(), directory)
    
    # Remove directory if it exists
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    # Create fresh directory
    os.makedirs(base_path)
    
    return base_path

def create_property_graph(property_graph_path: str):
    """
    Create property graph from documents in temp directory and save visualization
    """
    try:
        # Get the temp directory path
        temp_dir = os.path.join(os.getcwd(), "temp_text_for_property")
        
        # Load documents from temp directory
        documents = SimpleDirectoryReader(temp_dir).load_data()

        # Create property graph index
        index = PropertyGraphIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            show_progress=True
        )

        # Save graph visualization as HTML
        index.property_graph_store.save_networkx_graph(property_graph_path)
        print(f"Property graph visualization saved to {property_graph_path}")

        return index

    except Exception as e:
        print(f"Error creating property graph: {str(e)}")
        return None

@router.post("/extract_md")
async def extract_markdown():
    """
    Extract markdown content from the JSON file and save to temporary directory
    """
    try:
        # Get paths
        json_path = os.path.join(os.getcwd(), "..", "client", "public", "file-sample_150.json")
        property_graph_path = os.path.join(os.getcwd(), "..", "client", "public", "property_graph.html")
        # Read JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create/clear temp directory
        temp_dir = ensure_temp_directory()
        
        # Extract markdown from each page
        for item in data:
            if "pages" in item:
                for page in item["pages"]:
                    if "md" in page:
                        # Create filename for this page
                        filename = f"page_{page['page']}.txt"
                        file_path = os.path.join(temp_dir, filename)
                        # Write markdown content to file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(page["md"])

        # Print to console
        print(f"Markdown content extracted to {temp_dir}")

        # Create property graph and save visualization
        index = create_property_graph(property_graph_path)
        
        return {
            "message": "Markdown content extracted and property graph visualization created successfully",
            "temp_directory": temp_dir,
            "graph_visualization": os.path.join(temp_dir, "property_graph.html")
        }
    except Exception as e:
        return {"error": str(e)}
