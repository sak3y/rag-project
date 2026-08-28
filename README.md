# RAG Pipeline

A retrieval augmented generation pipeline that runs fully on your own machine by default. Documents are chunked, embedded with a local model served by Ollama, stored in a local Chroma vector database, then retrieved at query time and passed as context to the LLM.

Generation defaults to a local Ollama model (`llama3.2`). Because the LLM is created through `init_chat_model`, you can swap in a hosted provider like Gemini with two environment variables, no code changes. Embeddings always run locally through Ollama, so the documents themselves never leave this machine.

The test corpus is my personal notes, but the pipeline is corpus-agnostic. Drop any `.pdf`, `.md`, or `.txt` files into the docs folder.

## Architecture

```
docs -> loading -> chunking -> embedding -> vector store (Chroma) -> retrieval -> generation
```

## Design decisions

**Chunking.** `RecursiveCharacterTextSplitter` with `chunk_size=800` characters and `chunk_overlap=120`. That's roughly 160 to 200 tokens per chunk: big enough to hold one coherent idea, small enough that the embedding stays sharp instead of averaging several topics together. The overlap repeats a slice of text between adjacent chunks so an idea split across a boundary survives intact in at least one of them. Both values are env-configurable; the right size depends on how dense and how fragmented the corpus is.

**Deterministic chunk IDs.** Each chunk gets an md5 ID derived from its source path and position, so re-running the indexer upserts instead of duplicating. Pass `rebuild=True` to `index()` to wipe and re-embed from scratch.

**Vector store.** Chroma, persisted locally to `chroma_db/`. No external service required.

**Why Ollama for embeddings instead of `HuggingFaceEmbeddings`.** The LangChain HuggingFace integration pulls in `torch` (roughly 2 GB, slow to install, version-sensitive). Ollama serves the embedding model over HTTP, so the Python environment only needs a client.

**Why Python 3.12.** Compiled packages like Chroma and torch ship prebuilt wheels for new Python versions months late. On the newest interpreter, pip falls back to compiling from source, which is slow and fails often.

## Setup

Run once in a terminal, not inside the notebook:

```bash
brew install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install langchain langchain-core langchain-community langchain-text-splitters \
               langchain-chroma langchain-ollama langchain-google-genai pypdf python-dotenv ipykernel
ollama pull nomic-embed-text
ollama pull llama3.2
```

Then select `.venv` as the kernel (top right in VS Code).

## Configuration

Everything is overridable via environment variables (a `.env` file is loaded automatically):

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DOC_DIR` | `./docs` | Folder scanned recursively for documents |
| `RAG_DB_DIR` | `./chroma_db` | Chroma persistence directory |
| `RAG_CHUNK_SIZE` | `800` | Chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `120` | Overlap between adjacent chunks, in characters |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `RAG_LLM` | `llama3.2` | Generation model name |
| `RAG_LLM_PROVIDER` | `ollama` | Provider passed to `init_chat_model` |

To generate with Gemini instead of a local model:

```
RAG_LLM_PROVIDER=google_genai
RAG_LLM=gemini-2.5-flash
GOOGLE_API_KEY=your-key-here
```

## Usage

```python
store, n_chunks, n_files = index()          # or index(rebuild=True)
answer, hits = ask(store, "What did I write about chunking?")

# streaming
for token in stream(store, "Summarize my notes on Chroma"):
    print(token, end="", flush=True)
```

Retrieval pulls the top `k=4` chunks by similarity. The prompt instructs the model to answer only from the retrieved context, cite the `[source]` it used, and admit when the context doesn't contain the answer.

## Credits

Built following LangChain's [RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) video series, adapted for local embeddings and my own corpus.
