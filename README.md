# PDF RAG Application Server

This is the backend server for the PDF RAG (Retrieval-Augmented Generation) Application. It handles PDF processing, document analysis, and intelligent question-answering capabilities.

## Features

- 📄 PDF Processing & Analysis
- 🔍 Text Extraction & Chunking
- 🧠 Intelligent Question Answering
- 🔗 Vector Database Integration
- 🚀 Fast API Endpoints
- 🔒 Secure File Handling

## Tech Stack

- **Framework**: FastAPI
- **Database**: Pinecone (Vector Database)
- **AI Models**: 
  - OpenAI GPT-3.5/4 for text generation
  - OpenAI Ada-002 for embeddings
- **PDF Processing**: PyPDF2, pdf2image

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/MinhThieu145/pdf-rag-application-server.git
cd pdf-rag-application-server
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

5. **Run the server**
```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Main Endpoints

- `POST /upload`: Upload and process PDF documents
- `POST /chat`: Get answers to questions about the documents
- `GET /documents`: List processed documents
- `GET /health`: Server health check

## Project Structure

```
server/
├── config/         # Configuration files
├── controllers/    # Request handlers
├── helpers/        # Utility functions
├── lib/           # Core business logic
├── middleware/    # Request/response middleware
├── models/        # Data models
├── schemas/       # Pydantic schemas
└── main.py        # Application entry point
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|-----------|
| OPENAI_API_KEY | OpenAI API key | Yes |
| PINECONE_API_KEY | Pinecone API key | Yes |
| PINECONE_ENVIRONMENT | Pinecone environment | Yes |

## Error Handling

The server implements comprehensive error handling:
- Input validation errors
- Processing errors
- Authentication errors
- Rate limiting

## Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes

### Testing
```bash
pytest tests/
```

### Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Production Deployment

1. Set up proper CORS configuration
2. Configure rate limiting
3. Set up monitoring
4. Use production WSGI server (e.g., Gunicorn)
5. Set up proper logging

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Related Projects

- [PDF RAG Application Frontend](https://github.com/MinhThieu145/pdf-rag-application)

---
Made with ❤️ by MinhThieu145
