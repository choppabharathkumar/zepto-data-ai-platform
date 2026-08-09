# Dockerfile — Zepto Support Assistant
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Run document ingestion to build the ChromaDB vector store
RUN python ingest.py

# Expose FastAPI port
EXPOSE 8000

# Start the API server
# Set MOCK_LLM=1 to run without a real LLM API key
# Set GOOGLE_API_KEY for production use with Gemini
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
