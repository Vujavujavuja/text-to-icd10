# ICD-10 Coding Assistant

FastAPI backend for mapping clinical text to ICD-10 codes using hierarchy-aware RAG with optional LLM enhancement.

## Features

- **Semantic code matching** using FAISS vector search (72,000 ICD-10 codes)
- **Hierarchical validation** using ICD-10 chapter structure
- **LLM-enhanced clinical endpoint** for entity extraction, documentation gap detection, and explainable suggestions
- **Fast startup** with pre-computed embeddings (~2-3 seconds)

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your OpenRouter API key (get free key at https://openrouter.ai/keys):

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LLM_MODEL=anthropic/claude-haiku-4.5
LLM_ENABLED=true
```

### 3. Start Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

API available at: http://localhost:8080
Interactive docs: http://localhost:8080/docs

## API Endpoints

### POST /suggest - Basic Text Mapping

```bash
curl -X POST http://localhost:8080/suggest \
  -H "Content-Type: application/json" \
  -d '{"text": "Type 2 diabetes with foot ulcer"}'
```

### POST /suggest/clinical - LLM-Enhanced Clinical Coding

```bash
curl -X POST http://localhost:8080/suggest/clinical \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_notes": "Patient is a 67-year-old male with type 2 diabetes presenting with foot ulcer on left heel.",
    "enable_llm_extraction": true,
    "enable_llm_explanations": true
  }'
```

Response includes:
- `results`: Ranked ICD-10 codes with confidence scores and explanations
- `extracted_entities`: Symptoms, anatomical sites, laterality, severity
- `documentation_gaps`: Missing specificity for accurate coding

### GET /health - Service Health Check

```bash
curl http://localhost:8080/health
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM features | None |
| `LLM_MODEL` | Model to use | `anthropic/claude-haiku-4.5` |
| `LLM_ENABLED` | Enable LLM enhancement | `false` |
| `TOP_K` | Number of results to return | `5` |
| `MIN_CONFIDENCE_SCORE` | Minimum confidence threshold | `0.5` |

## Project Structure

```
text-to-icd10/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration
│   ├── models/schemas.py       # Request/response models
│   ├── services/
│   │   ├── llm_client.py       # OpenRouter LLM client
│   │   ├── llm_service.py      # Clinical extraction service
│   │   ├── retrieval_service.py # RAG retrieval
│   │   └── ...
│   └── api/routes.py           # API endpoints
├── data/cache/                 # Pre-computed embeddings
│   ├── enriched_dataset.pkl
│   ├── icd10_index.faiss
│   └── metadata.json
├── tests/                      # Test files
├── notebooks/                  # Preprocessing notebook
└── requirements.txt
```

## Running Tests

```bash
pytest tests/
```

## Performance

- **Server startup**: ~2-3 seconds
- **Basic query**: <100ms
- **LLM-enhanced query**: 2-5 seconds

