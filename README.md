# 📚 Local History RAG System

A **Retrieval-Augmented Generation (RAG)** system that answers historical questions by retrieving relevant context from source documents and generating answers using an LLM.

## 🎯 What is RAG?

RAG combines two powerful capabilities:
1. **Retrieval** - Finds relevant document chunks based on semantic similarity
2. **Generation** - Uses an LLM to generate answers grounded in retrieved context

This approach improves accuracy and allows the system to cite sources.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  User Question                          │
└────────────────┬────────────────────────┘
                 │
        ┌────────▼─────────┐
        │  Vector Search   │  (semantic similarity)
        └────────┬─────────┘
                 │
        ┌────────▼──────────────────┐
        │  Retrieved Chunks + Scores│
        └────────┬──────────────────┘
                 │
        ┌────────▼──────────────────┐
        │  Combine Context          │
        └────────┬──────────────────┘
                 │
        ┌────────▼──────────────────┐
        │  LLM Generation           │
        └────────┬──────────────────┘
                 │
        ┌────────▼──────────────────┐
        │  Answer with Sources      │
        └──────────────────────────┘
```

## 📁 Project Structure

```
rag-local/
├── main.py                      # Query interface (imports from src/)
├── ingest.py                    # Multi-modal document ingestion script
├── docker-compose.yml           # Neo4j container setup
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── .env.example                 # Configuration template
├── resources/
│   └── history.txt              # Source document
├── logs/                        # Application logs
│
├── src/                         # Core application code
│   ├── config.py                # Configuration management
│   ├── logger_config.py         # Logging setup
│   │
│   ├── core/
│   │   ├── pipeline.py          # RAG pipeline orchestration
│   │   └── stepback.py          # Step-back prompting
│   │
│   ├── models/                  # ML model providers
│   │   ├── embedding/           # Text → vector
│   │   │   ├── base.py          # Abstract EmbedderBase class
│   │   │   └── ollama.py        # OllamaEmbedder implementation
│   │   ├── llm/                 # Text → text
│   │   │   ├── base.py          # Abstract LLMBase class
│   │   │   └── ollama.py        # OllamaLLM implementation
│   │   └── vision/              # Image → text
│   │       ├── base.py          # Abstract VisionModelBase class
│   │       └── ollama.py        # OllamaVisionModel (LLaVA) implementation
│   │
│   ├── processing/              # Multi-modal content processing
│   │   ├── base.py              # ProcessorBase, ProcessedChunk, ContentType
│   │   ├── router.py            # ContentRouter (MIME detection & routing)
│   │   ├── text.py              # Text file processor
│   │   ├── image.py             # Image processor (via vision model)
│   │   └── pdf.py               # PDF processor (text + image extraction)
│   │
│   ├── db/                      # Database infrastructure
│   │   └── database.py          # Neo4j driver & index management
│   │
│   ├── ingestion/               # Document storage
│   │   └── ingestion.py         # Document chunking & storage in Neo4j
│   │
│   ├── retrieval/               # Document retrieval
│   │   └── vector_search.py     # Semantic similarity search
│   │
│   ├── schemas/
│   │   └── schemas.py           # Data schemas (SearchResult, Question, Answer)
│   │
│   └── utils/
│       └── exceptions.py        # Custom exception hierarchy
│
└── tests/                       # Comprehensive test suite
    ├── conftest.py              # pytest fixtures
    ├── test_core.py             # Pipeline validation tests
    ├── test_embedding.py        # Embedding provider tests
    ├── test_llm.py              # LLM provider tests
    ├── test_retrieval.py        # Chunking & search tests
    └── test_models.py           # Schema validation tests
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Docker (for Neo4j)
- Required Python packages (see below)

### Setup

1. **Start Neo4j Database:**
   ```bash
   docker-compose up -d
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ingest Documents (one-time):**
   ```bash
   python ingest.py
   ```
   This loads your document from `resources/history.txt`, chunks it, and stores embeddings in Neo4j.

4. **Run Queries:**
   ```bash
   python main.py
   ```
   This starts the interactive query interface. Questions are answered using the ingested document.

## 💬 Usage

### Workflow

The RAG system operates in two separate phases:

**Phase 1: Document Ingestion (one-time setup)**
```bash
python ingest.py
```

Supports `--file`/`--directory` for target selection, `--strategy` (`fixed` or `parent_child`), and `--split-method` (`fixed_size`, `title`, `tag`) for chunking control.

```bash
python ingest.py --file document.pdf                             # Ingest single file
python ingest.py --directory ./docs                              # Ingest all files in directory
python ingest.py --strategy parent_child --split-method title    # Parent-child with heading split
python ingest.py --directory ./docs --no-recursive               # Non-recursive directory scan
```

Output:
```
📚 RAG Document Ingestion Tool
============================================================

📖 Loading document from: resources/history.txt
✅ Document loaded (125403 characters)

🔄 Ingesting document into vector database...
✅ Ingestion complete!
============================================================
```

**Phase 2: Interactive Queries**
```bash
python main.py
```
Output:
```
📚 Local History RAG System
(Use 'ingest.py' to load documents)
Type 'exit' to quit.

Ask a question: When did Napoleon invade Russia?

🤖 Answer:
Napoleon invaded Russia in 1812...
============================================================

Ask a question: exit
Goodbye 👋
```

### How It Works

Once the system is running, you'll:
1. **Ask** a natural language question
2. **Retrieve** relevant historical chunks with similarity scores
3. **Display** the context sources
4. **Generate** an answer using the LLM
5. **Show** the final answer

### Example Questions

- When was Napoleon born?
- When did Napoleon invade Russia?
- What happened in 1815 to Napoleon?
- When did World War I begin?
- When did World War II end?
- When did the Cold War end?
- Who invaded Poland in 1939?
- What event triggered World War I?

## 🔧 Core Components

### `ingest.py`
- Standalone document ingestion script
- Loads document from `DOCUMENT_PATH` (configured via .env)
- Handles errors gracefully with detailed logging
- Can be run once after setup or on a schedule
- No external dependencies beyond src modules

### `main.py`
- Query-only CLI interface for the RAG system
- Awaits ingested data from Neo4j
- Orchestrates the repeating query loop
- Imports core pipeline from `src.core.pipeline`
- Graceful error handling with user-friendly messages

### `src/core/pipeline.py`
- **`rag_pipeline(question: str) -> str`** - Main RAG orchestration
  - Validates input question
  - Retrieves relevant chunks
  - Combines context
  - Generates answer using LLM
- **`validate_question(question: str) -> Tuple[bool, str]`** - Input validation

### `src/models/embedding/`
- **Base Class:** `EmbedderBase` (abstract)
  - Enables multiple embedding provider implementations
  - `embed(text: str) -> List[float]`
  - `get_model_info() -> dict`
- **Implementation:** `OllamaEmbedder`
  - Uses Ollama API for embeddings
  - 3-retry logic with 30s timeout
  - Singleton pattern: `get_embedder()`
  - Backward compatible: `embed_text()` function

### `src/models/llm/`
- **Base Class:** `LLMBase` (abstract)
  - Enables multiple LLM provider implementations
  - `generate(context: str, question: str) -> str`
  - `get_model_info() -> dict`
- **Implementation:** `OllamaLLM`
  - Uses Ollama API for text generation
  - 2-retry logic with 120s timeout
  - Singleton pattern: `get_llm()`
  - Backward compatible: `generate_answer()` function

### `src/models/vision/`
- **Base Class:** `VisionModelBase` (abstract)
  - Enables multiple vision model implementations
  - `describe(image_path) -> str`
  - `describe_bytes(image_bytes) -> str`
- **Implementation:** `OllamaVisionModel`
  - Uses Ollama LLaVA for image descriptions
  - Singleton pattern: `get_vision_model()`
  - Convenience function: `describe_image()`

### `src/db/database.py`
- **Driver Management:**
  - `get_driver()` - Lazy-loaded Neo4j driver singleton
  - `close_driver()` - Cleanup (registered with atexit)
- **Vector Index:**
  - `vector_index_exists(index_name) -> bool`
  - `create_vector_index(index_name) -> bool` - Idempotent creation

### `src/ingestion/ingestion.py`
- **`store_processed_chunk(tx, chunk_id, chunk)`** - Store a ProcessedChunk in Neo4j
- **`ingest_file(file_path, start_id) -> int`** - Ingest a single file via ContentRouter
- **`ingest_directory(directory, start_id) -> int`** - Ingest all supported files in a directory

### `src/retrieval/vector_search.py`
- **`retrieve(question: str, top_k: int) -> List[Tuple[str, float]]`**
  - Embeds user question
  - Performs vector similarity search
  - Returns top-k chunks with scores (0-1 cosine similarity)

### `src/schemas/schemas.py`
Data models with automatic validation:
- **`SearchResult`** - Chunk with similarity score
- **`Question`** - Validated user input (3-1000 chars)
- **`Answer`** - Generated answer with sources
- **`EmbeddingResult`** - Embedding operation result
- **`DatabaseConfig`** - Database connection settings

### `src/config.py`
Centralized configuration from environment variables:
- **Database:** NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
- **Ollama:** OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL, OLLAMA_LLM_MODEL
- **Vision:** OLLAMA_VISION_MODEL, VISION_TIMEOUT (120s)
- **Timeouts:** EMBEDDING_TIMEOUT (30s), LLM_TIMEOUT (120s)
- **Validation:** MIN_QUESTION_LENGTH (3), MAX_QUESTION_LENGTH (1000)
- **Ingestion:** CHUNK_SIZE (300), CHUNK_OVERLAP (200)

### `src/logger_config.py`
- Structured logging (file + console)
- Configurable log levels
- Auto-creates logs/ directory

### `src/utils/exceptions.py`
Custom exception hierarchy:
- `RAGException` (base)
- `EmbeddingError` - Embedding failures
- `LLMError` - LLM generation failures
- `DatabaseError` - Database operation failures
- `ValidationError` - Input validation failures
- `RetrievalError` - Search operation failures

## 📊 Data Flow

### Ingestion Flow (ingest.py):

Supports `--file`/`--directory` for target selection, `--strategy` (`fixed` or `parent_child`), and `--split-method` (`fixed_size`, `title`, `tag`) for chunking control.

```
ingest.py
    ↓
Load file/directory path from CLI args or DOCUMENT_PATH
    ↓
ingest_file() / ingest_directory()          # src/ingestion/
    ↓
ContentRouter → selects TextProcessor, ImageProcessor, or PDFProcessor
    ↓
Processor.process() → chunks content into ProcessedChunk objects
    ↓
create_vector_index() → creates Neo4j vector index (idempotent)  # src/db/
    ↓
for each chunk:
    embed_text() → OllamaEmbedder.embed() → Ollama API (3 retries)  # src/models/embedding/
        ↓
    store_processed_chunk() → Neo4j CREATE (Chunk nodes with embeddings)
    ↓
✅ Database ready for queries
```

### Query Flow (main.py):
```
User: "When did Napoleon invade Russia?"
    ↓
validate_question() → checks length, characters, format
    ↓
rag_pipeline() orchestrates:
    ├─ embed_text(question) → OllamaEmbedder          # src/models/embedding/
    │   └─ Ollama API (with retry logic)
    │
    ├─ retrieve(question) → vector_search.py            # src/retrieval/
    │   └─ db.index.vector.queryNodes() with cosine similarity
    │
    ├─ combine_context() → format sources with scores
    │
    └─ generate_answer(context, question) → OllamaLLM   # src/models/llm/
        └─ Ollama API (with retry logic & 120s timeout)
    ↓
Answer: "Napoleon invaded Russia in 1812..."
```

## 🗄️ Database

**Neo4j** stores:
- Document chunks
- Embeddings for semantic search
- Metadata and relationships
- System configuration

Data location: `../data/databases/neo4j/`

## 🧪 Testing

The project includes comprehensive unit tests with mocks:

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=src

# Run single test
pytest tests/test_core.py::TestCoreValidation::test_validate_question_valid -v
```

**Test Coverage:**
- `test_core.py` - Pipeline validation (empty, short, invalid inputs)
- `test_embedding.py` - Embedding provider (success, errors, empty text)
- `test_llm.py` - LLM provider (success, empty context, errors)
- `test_retrieval.py` - Text chunking and search (basic, empty, small)
- `test_models.py` - Schema validation (valid data, type errors, ranges)

## 🔌 Extensibility

The system uses abstract base classes for extensibility:

### Adding a New Embedder (e.g., OpenAI):

```python
# src/models/embedding/openai.py
from src.models.embedding.base import EmbedderBase

class OpenAIEmbedder(EmbedderBase):
    def embed(self, text: str) -> List[float]:
        # OpenAI API call
        pass
    
    def get_model_info(self) -> dict:
        return {"provider": "openai", "model": "text-embedding-3"}

# Update src/models/embedding/__init__.py to export
```

### Adding a New LLM (e.g., GPT-4):

```python
# src/models/llm/openai.py
from src.models.llm.base import LLMBase

class OpenAILLM(LLMBase):
    def generate(self, context: str, question: str) -> str:
        # OpenAI API call
        pass
    
    def get_model_info(self) -> dict:
        return {"provider": "openai", "model": "gpt-4"}

# Update src/models/llm/__init__.py to export
```

**Benefits:**
- No changes to core RAG pipeline
- Pluggable providers
- Easy testing with mocks
- Type-safe implementations

## ⚙️ Configuration

All settings are environment-driven via `.env`:

```env
# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Ollama Service
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=qwen3.5:9b
OLLAMA_TEMPERATURE=0.2

# Network Timeouts
EMBEDDING_TIMEOUT=30
LLM_TIMEOUT=120
DATABASE_TIMEOUT=30
RETRY_DELAY=1

# Ingestion
CHUNK_SIZE=300
CHUNK_OVERLAP=200
VECTOR_INDEX_NAME=chunk_embedding_index

# Validation
MIN_QUESTION_LENGTH=3
MAX_QUESTION_LENGTH=1000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag_system.log
DEBUG_MODE=false
```

See `.env.example` for complete template.

## 🛡️ Error Handling

The system uses custom exceptions for specific error types:

```python
try:
    answer = rag_pipeline(question)
except EmbeddingError:
    # Handle embedding API failures
    print("Embedding service unavailable")
except LLMError:
    # Handle LLM API failures
    print("LLM service unavailable")
except ValidationError:
    # Handle invalid input
    print("Invalid question format")
except DatabaseError:
    # Handle database failures
    print("Database connection failed")
```

**Features:**
- Automatic retry logic (3x for embeddings, 2x for LLM)
- Configurable timeouts
- Graceful degradation with user-friendly messages
- Structured error logging

## 🐛 Troubleshooting

### Common Issues:

| Issue | Solution |
|-------|----------|
| `Connection refused` | Ensure Neo4j is running: `docker-compose up -d` |
| `history.txt not found` | Create `resources/history.txt` with historical content |
| `Module not found: requests` | Install dependencies: `pip install -r requirements.txt` |
| `OLLAMA connection error` | Start Ollama: `ollama serve` (separate terminal) |
| `Vector index already exists` | The system handles this gracefully (idempotent) |
| `Ingestion takes too long` | Reduce CHUNK_SIZE in .env or use smaller documents |
| `Port 7687 in use` | Change NEO4J_BOLT_PORT in docker-compose.yml |
| `ImportError from src.*` | Ensure you run scripts from the rag-local directory |
| `main.py says "No chunks found"` | Run `ingest.py` first to load documents into database |
| `No database connection during ingest` | Verify Neo4j is healthy: `docker-compose ps` |

### Debug Mode:

Enable debug logging to see detailed operation flow:

```bash
DEBUG_MODE=true LOG_LEVEL=DEBUG python main.py
```

This will:
- Print detailed logs to console
- Save comprehensive logs to `logs/rag_system.log`
- Show retry attempts and timeouts
- Display embedding dimensions and search scores

### Performance Tuning:

| Parameter | Adjust For |
|-----------|-----------|
| CHUNK_SIZE | Larger chunks (500+) = fewer requests, less precise; Smaller chunks (200) = more requests, more precise |
| CHUNK_OVERLAP | Larger overlap (300+) = slower ingestion, better context continuity |
| SEARCH_TOP_K | Fewer results (3) = faster, less context; More results (10+) = slower, more context |
| LLM_TIMEOUT | Slow responses = increase to 180+; Fast API = decrease to 60 |

## 📦 Dependencies

Core dependencies:
- `neo4j>=5.14.0` - Graph database driver
- `requests>=2.31.0` - HTTP client for Ollama API
- `python-dotenv>=1.0.0` - Environment variable management
- `pytest>=7.0` - Testing framework (development only)

Install with:
```bash
pip install -r requirements.txt
```

For development/testing:
```bash
pip install pytest pytest-cov
```

## 🏗️ Architecture Highlights

### Design Patterns Used:

1. **Singleton Pattern** - Global driver and provider instances
   ```python
   driver = get_driver()  # Always returns same instance
   embedder = get_embedder()  # Always returns same instance
   ```

2. **Abstract Factory Pattern** - Provider extensibility
   ```python
   class EmbedderBase(ABC):
       @abstractmethod
       def embed(self, text: str) -> List[float]: pass
   ```

3. **Pipeline Pattern** - RAG orchestration
   ```python
   def rag_pipeline(question: str) -> str:
       # Validate → Retrieve → Generate
   ```

4. **Dataclass Validation** - Type-safe data models
   ```python
   @dataclass
   class SearchResult:
       text: str
       score: float  # Auto-validates range 0-1
   ```

### Quality Metrics:

- **Type Coverage:** 100% (full type hints)
- **Error Handling:** Comprehensive exception hierarchy  
- **Retry Logic:** Automatic with exponential backoff
- **Test Coverage:** 20+ unit tests
- **Code Organization:** 10 focused modules with clear responsibilities

## 🎓 How It Works

### Step 1: Vector Embeddings
Text is converted to semantic vectors using an embedding model:
```
"Napoleon invaded Russia in 1812" 
    → [0.12, -0.45, 0.89, ..., 0.33] (1536-dimensional)
```

### Step 2: Similarity Search
Question embedding is compared with chunk embeddings using cosine similarity:
```
Question: "When did Napoleon invade Russia?"
    → [0.15, -0.42, 0.87, ..., 0.31]
    
Cosine Similarity with chunk:
    (Q · C) / (||Q|| * ||C||) = 0.94  ← Very similar!
```

### Step 3: Vector Index (Neo4j)
Chunks are stored in a Neo4j VECTOR INDEX:
```cypher
CREATE VECTOR INDEX chunk_embedding_index
FOR (c:Chunk) ON (c.embedding)
OPTIONS { indexConfig: { 'vector.similarity_function': 'cosine' } }
```

### Step 4: Context Ranking
Top-k most similar chunks are retrieved with scores:
```
Chunk 1 (score=0.94): "In 1812, Napoleon launched Operation Barbarossa..."
Chunk 2 (score=0.89): "The invasion of Russia proved to be a turning point..."
Chunk 3 (score=0.87): "Russian winter and supply lines were Napoleon's downfall..."
```

### Step 5: Context-Aware Generation
LLM receives context + question and generates grounded answer:
```
Context: [top chunks with scores]
Question: "When did Napoleon invade Russia?"
LLM Response: "Napoleon invaded Russia in 1812, launching Operation 
Barbarossa. This invasion proved to be a major turning point..."
```

### Key Technologies:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | Ollama (nomic-embed-text) | Convert text to vectors |
| **LLM** | Ollama (qwen3.5:9b) | Generate contextual answers |
| **Vector DB** | Neo4j 5.14+ | Store and search embeddings |
| **Runtime** | Python 3.8+ | System orchestration |

## 📥 Import Structure

All source code uses `src.*` imports for consistency:

```python
# In any module:
from src.config import OLLAMA_BASE_URL, LLM_TIMEOUT
from src.logger_config import get_logger
from src.core.pipeline import rag_pipeline
from src.models.embedding.ollama import embed_text
from src.models.llm.ollama import generate_answer
from src.db.database import get_driver
from src.schemas.schemas import SearchResult, Answer
from src.utils.exceptions import EmbeddingError

logger = get_logger("my_module")
```

The `main.py` entry point handles sys.path:
```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

This allows clean imports throughout the codebase.

## 📚 Resources

- [RAG Concepts](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Vector Embeddings](https://en.wikipedia.org/wiki/Word_embedding)

## 📄 License

[Add your license here]

---

**Happy Exploring!** 🚀
