# Legislation RAG Pipeline

A retrieval augmented generation service over UK criminal law. Acts are fetched from the legislation.gov.uk API as XML, chunked along the law's own structural boundaries, embedded locally, and served over HTTP so an LLM answers from the statute with a citation rather than from memory.

## Architecture

```
legislation.gov.uk API  ->  XML parse  ->  one chunk per subsection
                                                |
                                     embed (nomic-embed-text, local)
                                                |
                                         Chroma vector store
                                                |
                               retrieve top k=4 nearest chunks
                                                |
                           LLM answers from those only  ->  FastAPI
```

Two modules. `rag.py` is the pipeline: fetch, chunk, embed, retrieve, generate. `api.py` is the HTTP layer and holds no pipeline logic.

**Corpus:** Theft Act 1968, Criminal Damage Act 1971, Misuse of Drugs Act 1971, Fraud Act 2006, Computer Misuse Act 1990. 568 subsections. Adding an Act is one line in `ACT_IDS`.

## Running it

Docker Compose runs the API and Ollama as two services on a shared network:

```bash
docker compose up -d

docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
docker compose exec api python rag.py
```

The last three are one-time. Models and the vector index both live in named volumes, so restarts are instant and nothing is re-downloaded or re-embedded. The API is then on `localhost:8000`, with interactive docs at `/docs`.

### Without Docker

```bash
uv venv --python 3.12 && source .venv/bin/activate    # .venv\Scripts\Activate.ps1 on Windows
uv pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull llama3.2

python rag.py            # builds the index, then runs the eval
uvicorn api:app --reload
```

Ollama must be running before anything else, and a fresh clone has no `chroma_db/`, so `python rag.py` is required before the API returns anything. Python is pinned to 3.12 because compiled packages ship prebuilt wheels for new interpreters months late, and pip otherwise falls back to compiling from source.

## API

`/search` is retrieval only: no model call, no cost, fast. `/ask` adds generation and is rate limited.

```bash
curl "localhost:8000/ask?query=what+is+burglary"
```

```json
{
  "query": "what is burglary",
  "answer": "According to Section 9(1), a person is guilty of burglary if he enters
             any building or part of a building as a trespasser with intent to commit
             an offence mentioned in subsection (2).",
  "sources": [
    {
      "text": "Section 9(1): A person is guilty of burglary if— he enters any building...",
      "uri": "http://www.legislation.gov.uk/ukpga/1968/60/section/9/1",
      "score": 0.603
    }
  ]
}
```

Sources come back with every answer, each carrying the URI it was built from, so any claim can be checked against legislation.gov.uk rather than taken on trust. `/health` reports the chunk count, which is how a missing or empty index announces itself instead of silently returning nothing.

## Design decisions

**Chunk by subsection, not character count.** A character splitter cuts mid-clause, leaving half a legal condition in each chunk and two unmatchable embeddings. The XML already marks real boundaries, so the loader chunks on those and every chunk is one complete provision.

**Citation baked into the embedded text.** Each chunk reads `Section 9(1): A person is guilty of burglary if...`. The identity is embedded with the content, so a subsection retrieved alone still reads as a citation. Metadata alone would not do this, since metadata is never embedded and the model only sees the text.

**No overlap.** Overlap exists to rescue ideas cut at arbitrary boundaries. There are no arbitrary boundaries here, so it buys nothing.

**IDs derived, not read.** Section plus subsection number is not unique, because Schedules repeat the body's numbering. The chunk ID is an md5 of the provision's URI, built from the parent element rather than read off the child, after finding a published Act where one child URI pointed at the wrong section.

**L2 distance, not cosine.** Cosine is the usual default for text because vector magnitude tracks document length. Every chunk here is one subsection, so lengths cluster and the two metrics rank near-identically. The default was kept rather than rebuilt for no measurable gain.

**Embeddings local, generation swappable.** Embedding runs on Ollama, so re-indexing is free and the corpus never leaves the machine. Generation goes through `init_chat_model`, making the provider configuration rather than code: two environment variables switch `llama3.2` for a hosted model, with the key in `.env` and no other change.

## Retrieval quality

Ten questions, each paired with the section that should answer it, scored on whether that section appears in the top k. The set lives in `tests/test.json` and runs with `evaluate(store)`.

**hit@4: 10/10.** Questions are phrased the way a member of the public would ask ("is hacking into a computer illegal", "is taking someone's car without permission a crime") rather than in statutory language, so the score reflects retrieval bridging plain English to legal drafting. An earlier corpus scored 9/10, with the failure traced to near-identical Schedule paragraphs crowding out the section that answered the question.

## Tests

Four pytest cases run against a committed XML fixture, so they need no network and no Ollama. One per input class: a plain subsection, a subsection with nested lettered points, a Schedule paragraph, and a shape check across every chunk asserting each has a unique ID and text behind its citation. Each pins a bug that actually occurred, which is the whole reason they exist.

```bash
python -m pytest -v
```

## Docker

Two services rather than one image. Ollama is pulled prebuilt and the API is built from the Dockerfile; Compose puts them on a shared network, so the API reaches the model at `http://ollama:11434` instead of localhost, which inside a container means the container itself. Both the models and the index sit in named volumes, because a container's own filesystem is discarded on restart.

The tradeoff: keeping the model containerised means the stack is free and entirely self-contained, but it is roughly 5 GB and wants several GB of RAM, so it will not run on a free hosting tier. The production alternative is hosted embeddings and generation, which makes the image small and deployable but costs per query. Free and self-contained was the deliberate choice.

## Working with the source data

The published XML is well structured but not clean. Five issues surfaced during development and each is handled in the loader: the API serves XML rather than JSON and must be parsed from raw bytes; every tag sits in an unprefixed namespace that silently returns nothing if omitted; section numbers can hide behind nested commentary tags; text quoted from other Acts appears inline and has to be excluded or it gets indexed as the wrong Act's law; and amended sections store their text in markup the obvious element search never reaches. A sixth is a genuine error in the published data, where one subsection carries another section's URI. Each was found by running the pipeline and reading the failure, not from documentation.

## Known gaps

- Sections without numbered subsections are skipped.
- The eval set is small and every question currently passes, which suggests it needs harder cases rather than that retrieval is perfect.
- No amendment tracking, so the corpus is a snapshot rather than current law.
- Vector search only. Exact-term lookups ("what does section 9 say") would benefit from hybrid keyword search.
- Not deployed. It runs locally under Compose but has no public instance.

## Credits

Structure follows LangChain's [RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) series. The loader, chunking strategy, and legislation-specific handling are original work against the [legislation.gov.uk API](https://www.legislation.gov.uk/developer).