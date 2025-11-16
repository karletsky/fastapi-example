# FastAPI RAG Chat Application

A modern FastAPI application with ChatGPT integration and RAG (Retrieval Augmented Generation) capabilities using Milvus vector database.

## Features

- 💬 **Interactive Chat Interface** - Beautiful, animated web UI for chatting with ChatGPT
- 📚 **Knowledge Base Management** - Add documents to enhance AI responses with your own knowledge
- 🔍 **RAG System** - Semantic search using Milvus vector database and sentence transformers
- 🎨 **Modern Design** - Glass morphism UI with smooth animations and gradients
- 🚀 **Real-time Responses** - Streaming responses with typing indicators

## Tech Stack

- **Backend**: FastAPI
- **AI**: OpenAI GPT-3.5-turbo
- **Vector DB**: Milvus (with Docker)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Frontend**: Vanilla JavaScript with modern CSS animations

## Prerequisites

- Python 3.11+
- Docker Desktop (for Milvus)
- OpenAI API key

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/karletsky/fastapi-example.git
cd fastapi-example
```

2. **Create virtual environment and install dependencies**
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

4. **Start Milvus with Docker Compose**
```bash
docker-compose up -d
```
Wait ~30 seconds for Milvus to fully initialize.

5. **Run the application**
```bash
uvicorn main:app --reload
```

## Usage

Open http://127.0.0.1:8000/ in your browser.

### Chat with AI
- Type your message in the chat input on the left
- Press Enter or click "Send"
- The AI will respond using ChatGPT, enhanced with any relevant knowledge from your database

### Add Knowledge to RAG
- Paste documents or text into the right panel
- Click "Add to Knowledge Base"
- The text will be embedded and stored in Milvus
- Future chat queries will use this knowledge as context

## API Endpoints

- `GET /` - Serves the web interface
- `POST /ask` - Chat with AI (with RAG context)
  ```json
  {"message": "your question"}
  ```
- `POST /ingest` - Add document to knowledge base
  ```json
  {"text": "your document content"}
  ```
- `GET /docs` - Interactive Swagger API documentation

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `MILVUS_HOST` | `localhost` | Milvus server host |
| `MILVUS_PORT` | `19530` | Milvus server port |

## Architecture

The application uses a RAG (Retrieval Augmented Generation) architecture:

1. Documents are converted to embeddings using sentence-transformers
2. Embeddings are stored in Milvus vector database
3. When a user asks a question:
   - The question is converted to an embedding
   - Top 3 most similar documents are retrieved from Milvus
   - Retrieved context is added to the ChatGPT prompt
   - ChatGPT generates a response using the context

## Stopping the Application

**Stop FastAPI server**: Press `Ctrl+C` in the terminal

**Stop Milvus**:
```bash
docker-compose down
```

## License

MIT
