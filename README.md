# RAG Pipeline

A retrieval augmented generation pipeline. Documents are chunked, embedded, and retrieved to give an LLM
grounded context.

The corpus is personal notes and experience, so everything runs locally.
Generation goes through Ollama and embeddings come from Hugging Face models,
which keeps the documents on this machine.

Design decisions are not settled yet. Chunking strategy, embedding model, and
vector store are all still being tested.


## Setup (run once in a terminal, not here)

```bash
brew install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install langchain langchain-core langchain-community langchain-text-splitters \
               langchain-chroma langchain-ollama langchain-google-genai pypdf python-dotenv ipykernel
ollama pull nomic-embed-text
```

Select `.venv` as the kernel (top right in VS Code).

Why 3.12: compiled packages (Chroma, torch) ship wheels for new Python versions months late. On the newest interpreter pip compiles from source, which is slow and fails often.

Why Ollama for embeddings: `HuggingFaceEmbeddings` pulls in `torch` (2 GB, slow, version-sensitive). Ollama runs the model as a local server, so Python only needs an HTTP client.

=======
## Setup
1. Python 3.12 (newer versions lack prebuilt wheels for chroma/torch)
2. `ollama pull nomic-embed-text`
3. Put documents (.md, .txt, .pdf) in `docs/`
4. GOOGLE_API_KEY in `.env` if using Gemini
5. Open `rag_pipeline.ipynb`, run top to bottom
>>>>>>
