from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import rag # connects to pipeline


# Config
app = FastAPI()
store = rag.get_store()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
def health():
    return {
        "chunks": store._collection.count()
    }

@app.get("/search")
def search(query: str, k_neighbours: int = 4):
    # find the source within the vectore store
    hits = rag.retrieve(store, query, k_neighbours)
    res = []
    for doc, score in hits:
        res.append({
            "text": doc.page_content,
            "uri": doc.metadata["uri"],
            "score": score
        })
    return {
        "query": query,
        "res": res
    }

@app.get("/ask")
@limiter.limit("3/minute")
def ask(request: Request, query: str, k: int = 4):
    answer, hits = rag.ask(store, query, k)
    return {"answer": answer}
