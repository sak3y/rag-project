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
    # find the source within the vectore store
    hits = rag.retrieve(store, question, k_neighbours)
    res = []
    for doc, score in hits:
        res.append({
            "text": doc.page_content,
            "uri": doc.metadata["uri"],
            "score": score
        })
    return {
        "query": question,
        "res": res
    }