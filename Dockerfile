FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (much smaller than full torch)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding model during build so it's cached in the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
