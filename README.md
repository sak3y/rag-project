# RAG Pipeline

A retrieval augmented generation pipeline that runs mostly on your own machine. Documents are chunked, embedded with a local model served by Ollama, and stored in a local Chroma vector database. At query time, relevant chunks are retrieved and passed as grounded context to the LLM.

Generation uses Gemini Flash via API key. An Ollama-hosted model can be swapped in as the generator instead, which keeps everything local if privacy is a concern. Embeddings always run locally through Ollama, so the documents themselves never leave this machine.

The test corpus is my personal notes, but the pipeline is corpus-agnostic. Point it at any set of documents.

## Architecture

```
docs -> splitting -> chunking -> indexing -> embedding -> vector store -> retrieval -> generation
```

## Design decisions

**Chunking strategy.** Currently targeting 80 to 120 tokens per chunk. The right size depends on the corpus: how the documents are formatted and how densely information is packed. Treat this as a tuning knob, not a constant.

**Vector store.** Chroma, running locally. No external service required.

**Why Ollama for embeddings instead of `HuggingFaceEmbeddings`.** The LangChain HuggingFace integration pulls in `torch` (roughly 2 GB, slow to install, version-sensitive). Ollama runs the embedding model as a local server, so the Python environment only needs an HTTP client.

**Why Python 3.12.** Compiled packages like Chroma and torch ship prebuilt wheels for new Python versions months late. On the newest interpreter, pip falls back to compiling from source, which is slow and fails often.

## Setup

Run once in a terminal, not inside the notebook:

```bash
brew install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install langchain langchain-core langchain-community langchain-text-splitters \
               langchain-chroma langchain-ollama langchain-google-genai pypdf python-dotenv ipykernel
ollama pull nomic-embed-text
```

Then select `.venv` as the kernel (top right in VS Code).

You'll also need a Google API key in a `.env` file for Gemini generation:

```
GOOGLE_API_KEY=your-key-here
```

## Credits

Built following LangChain's [RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) video series, adapted for local embeddings and my own corpus.
