import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Hello World API", version="0.1.0")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Milvus configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "knowledge_base"


def init_milvus():
    """Initialize Milvus connection and create collection if it doesn't exist."""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        
        # Check if collection exists
        if not utility.has_collection(COLLECTION_NAME):
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
            ]
            schema = CollectionSchema(fields=fields, description="Knowledge base for RAG")
            
            # Create collection
            collection = Collection(name=COLLECTION_NAME, schema=schema)
            
            # Create index for vector field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            print(f"Created collection: {COLLECTION_NAME}")
        else:
            collection = Collection(name=COLLECTION_NAME)
            collection.load()
            print(f"Loaded existing collection: {COLLECTION_NAME}")
            
    except Exception as e:
        print(f"Warning: Could not connect to Milvus: {e}")
        print("RAG features will be disabled. Make sure Milvus is running.")


# Initialize Milvus on startup
@app.on_event("startup")
async def startup_event():
    init_milvus()


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    response: str


class DocumentRequest(BaseModel):
    text: str


class DocumentResponse(BaseModel):
    message: str
    document_count: int


@app.get("/", summary="Serve index.html", tags=["static"])
def read_root():
    """Serve the main HTML page."""
    return FileResponse("index.html")


@app.post("/ingest", response_model=DocumentResponse, summary="Ingest Document", tags=["rag"])
async def ingest_document(request: DocumentRequest) -> DocumentResponse:
    """Add a document to the knowledge base."""
    try:
        # Ensure connection to Milvus
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        except:
            pass  # Connection might already exist
        
        # Generate embedding
        embedding = embedding_model.encode(request.text).tolist()
        
        # Get collection
        collection = Collection(name=COLLECTION_NAME)
        
        # Insert data
        collection.insert([[request.text], [embedding]])
        collection.flush()
        
        # Get total count
        collection.load()
        count = collection.num_entities
        
        return DocumentResponse(
            message="Document ingested successfully",
            document_count=count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus error: {str(e)}")


@app.post("/ask", response_model=MessageResponse, summary="Ask ChatGPT with RAG", tags=["chat"])
async def ask_chatgpt(request: MessageRequest) -> MessageResponse:
    """Send a message to ChatGPT with context from the knowledge base."""
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode(request.message).tolist()
        
        # Search for relevant context in Milvus
        context_text = ""
        try:
            # Ensure connection to Milvus
            try:
                connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            except:
                pass  # Connection might already exist
                
            collection = Collection(name=COLLECTION_NAME)
            collection.load()
            
            # Search for top 3 similar documents
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=3,
                output_fields=["text"]
            )
            
            # Extract relevant texts
            if results and len(results[0]) > 0:
                contexts = [hit.entity.get('text') for hit in results[0]]
                context_text = "\n\n".join(contexts)
        except Exception as milvus_error:
            # If Milvus fails, continue without context
            print(f"Milvus search failed: {milvus_error}")
        
        # Build the prompt with context
        system_message = "You are a helpful assistant."
        if context_text:
            system_message += f"\n\nUse the following context to answer questions:\n\n{context_text}"
        
        # Call OpenAI with context
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": request.message}
            ]
        )
        return MessageResponse(response=completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
