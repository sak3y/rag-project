from fastapi import FastAPI
import rag

app = FastAPI()
store = rag.get_store()

@app.get("/health")
def health():
    return {
        "chunks": store._collection.count()
    }

@app.get("/search")
def search(question: str, k_neighbours: int = 4):
    hits = rag.retrieve(store, question, k_neighbours)
    return {
        "query": question,
        "res": [d.page_content for d, _ in hits]
    }