import hashlib
import json
import os
import requests
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

# Config
DB_DIR = os.getenv("DB_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

ACT_IDS = [
    "ukpga/1968/60",   # Theft Act 1968
    "ukpga/1971/48",   # Criminal Damage Act 1971
    "ukpga/1971/38",   # Misuse of Drugs Act 1971
    "ukpga/2006/35",   # Fraud Act 2006
    "ukpga/1990/18",   # Computer Misuse Act 1990
]

ACT_URLS = [f"https://www.legislation.gov.uk/{a}/data.xml" for a in ACT_IDS]
NS = "{http://www.legislation.gov.uk/namespaces/legislation}"


def load(url):
    # Fetches one Act's XML and parses it into a searchable tree
    req = requests.get(url)
    if req.status_code != 200:
        return None
    return ET.fromstring(req.content)


def text_of(para):
    # Pulls out all the words, skipping the (a)/(b) numbering so it doesn't glue into the sentence
    parts = []
    for node in para.iter():
        if node.tag == f"{NS}Pnumber":
            continue
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        if node.tail and node.tail.strip():
            parts.append(node.tail.strip())
    return " ".join(parts)


def split(root, url):
    # Walks the Act and makes one chunk per subsection, each labelled with where it came from
    docs = []

    for p1 in root.iter(f"{NS}P1"):
        section = "".join(p1.find(f"{NS}Pnumber").itertext()).strip()
        p1_uri = p1.get("DocumentURI")

        # Schedules number their parts as paragraphs, not sections
        if p1_uri is not None and "/schedule/" in p1_uri:
            label = "Paragraph"
        else:
            label = "Section"

        for p2 in p1.iter(f"{NS}P2"):
            # No URI means this is text quoted from a different Act, so skip it
            if p2.get("DocumentURI") is None:
                continue

            sub = "".join(p2.find(f"{NS}Pnumber").itertext()).strip()
            content = text_of(p2.find(f"{NS}P2para"))

            # Built from the parent because some child URIs in the published XML are wrong
            uri = f"{p1_uri}/{sub}"

            docs.append(Document(
                page_content=f"{label} {section}({sub}): {content}",
                metadata={
                    "source": url,
                    "section": section,
                    "subsection": sub,
                    "uri": uri,
                    "type": label.lower(),
                },
            ))

    return docs


def chunk_id(doc):
    # each provision has the same ID every run, so re-indexing overwrites instead of adding copies
    return hashlib.md5(doc.metadata["uri"].encode()).hexdigest()


def get_store():
    # retrieves local vector store
    return Chroma(
        collection_name="legislation",
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST),
        persist_directory=DB_DIR,
    )


def index(urls=ACT_URLS, rebuild=False):
    # Fetches every Act, turns them into chunks, embeds them, and saves them to the database
    
    store = get_store()
    if rebuild:
        store.delete_collection()
        store = get_store()

    docs = []
    for url in urls:
        root = load(url)
        if root is None:
            print(f"failed to fetch: {url}")
            continue
        docs += split(root, url)

    # ran into issues sending the entire doc to ollama, send as batch instead
    batch = 100
    for i in range(0, len(docs), batch):
        part = docs[i:i + batch]
        store.add_documents(part, ids=[chunk_id(d) for d in part])
        print(f"  {i + len(part)}/{len(docs)}")

    store.add_documents(docs, ids=[chunk_id(d) for d in docs])
    print(f"{len(docs)} subsections indexed, store holds {get_store()._collection.count()}")
    return store


def retrieve(store, question, k=4):
    # Finds the k chunks whose meaning sits closest to the question
    return store.similarity_search_with_score(question, k=k)


_llm = None
def get_llm():
    # Built on first use, not on import, so retrieval-only callers never load a model
    global _llm
    if _llm is None:
       _llm = init_chat_model(LLM_MODEL, model_provider="ollama", base_url=OLLAMA_HOST)
    else:
        _llm = init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER)
    return _llm


def build_prompt(question, hits):
    # Glues the retrieved sections onto the question, with instructions to answer only from them
    context = "\n\n".join(d.page_content for d, _ in hits)
    return (f"Answer using only the context below. Cite the section you used. "
            f"If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}")


def ask(store, question, k=4):
    # Finds the closest sections, then asks the model to answer using only those.
    # Returns the hits as well so callers can show what the answer was built from.
    hits = retrieve(store, question, k)
    answer = get_llm().invoke(build_prompt(question, hits)).text
    return answer, hits


def evaluate(store, k=4, path="tests/test.json"):
    # Runs every test question and counts how often the right section came back in the top k
    cases = json.load(open(path))
    hits_at_k = 0
    for case in cases:
        # pass = any subsection of the expected section appears in the results
        results = retrieve(store, case["question"], k)
        found = any(d.metadata["uri"].startswith(case["expected"]) for d, _ in results)
        hits_at_k += found
        print(f"{'PASS' if found else 'FAIL'}  {case['question']}")
        if not found:
            print(f"        expected {case['note']}")
    print(f"\nhit@{k}: {hits_at_k}/{len(cases)}")
    return hits_at_k, len(cases)


if __name__ == "__main__":
    store = index(rebuild=False)
    evaluate(store)

    