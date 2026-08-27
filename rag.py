
import glob, hashlib, os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.getenv("RAG_DOC_DIR", os.path.join(ROOT, "docs"))
DB_DIR = os.getenv("RAG_DB_DIR", os.path.join(ROOT, "chroma_db"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 120))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("RAG_LLM", "llama3.2")
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "ollama")

LOADERS = {
    ".pdf": lambda p: PyPDFLoader(p),
    ".md":  lambda p: TextLoader(p, encoding="utf-8"),
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
}


def load_docs(folder=DOC_DIR):
    docs = []
    for path in sorted(glob.glob(f"{folder}/**/*", recursive=True)):
        ext = os.path.splitext(path)[1].lower()
        if ext in LOADERS:
            docs += LOADERS[ext](path).load()
    if not docs:
        raise SystemExit(f"no readable files under {folder}")
    return docs


def split(docs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(docs)


def chunk_id(doc, i):
    return hashlib.md5(f"{doc.metadata['source']}::{i}".encode()).hexdigest()


def get_store():
    return Chroma(collection_name="docs",
                  embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
                  persist_directory=DB_DIR)


def index(rebuild=False, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    store = get_store()
    if rebuild:
        store.delete_collection()
        store = get_store()
    chunks = split(load_docs(), size, overlap)
    store.add_documents(chunks, ids=[chunk_id(d, i) for i, d in enumerate(chunks)])
    return store, len(chunks), len({d.metadata["source"] for d in chunks})


def retrieve(store, question, k=4):
    return store.similarity_search_with_score(question, k=k)


def build_prompt(question, hits):
    ctx = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d, _ in hits)
    return (f"Answer using only the context below. Cite the [source] you used. "
            f"If the context does not contain the answer, say so.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}")


def get_llm():
    return init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER)


def ask(store, question, k=4, llm=None):
    llm = llm or get_llm()
    hits = retrieve(store, question, k)
    return llm.invoke(build_prompt(question, hits)).text, hits


def stream(store, question, k=4, llm=None):
    """Yield answer tokens as they arrive. Returns hits via StopIteration value."""
    llm = llm or get_llm()
    hits = retrieve(store, question, k)
    for chunk in llm.stream(build_prompt(question, hits)):
        yield chunk.text
    return hits