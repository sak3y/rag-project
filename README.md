# RAG Pipeline

A retrieval augmented generation pipeline. Documents are chunked, embedded, and retrieved to give an LLM
grounded context.

The corpus is personal notes and experience, so everything runs locally.
Generation goes through Ollama and embeddings come from Hugging Face models,
which keeps the documents on this machine.

Design decisions are not settled yet. Chunking strategy, embedding model, and
vector store are all still being tested.

## Setup
1. Python 3.12 (newer versions lack prebuilt wheels for chroma/torch)
2. `ollama pull nomic-embed-text`
3. Put documents (.md, .txt, .pdf) in `doc/`
4. GOOGLE_API_KEY in `.env` if using Gemini
5. Open `rag_pipeline.ipynb`, run top to bottom