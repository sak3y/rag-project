# Legislation RAG Pipeline

A retrieval augmented generation pipeline over UK criminal law. Acts are fetched from the legislation.gov.uk API as XML, chunked along the law's own structural boundaries, embedded locally, and retrieved at query time so an LLM answers from the statute with a citation rather than from memory.

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
                                   LLM answers from those only
```

Seven functions, one job each: `load` fetches and parses, `text_of` extracts prose, `split` chunks, `chunk_id` and `get_store` handle persistence, `index` builds the store, `retrieve` and `ask` serve queries.

**Corpus:** Theft Act 1968, Criminal Damage Act 1971, Misuse of Drugs Act 1971, Fraud Act 2006, Computer Misuse Act 1990. 568 subsections. Adding an Act is one line in `ACT_IDS`.

## Design decisions

**Chunk by subsection, not character count.** A character splitter cuts mid-clause, leaving half a legal condition in each chunk and two unmatchable embeddings. The XML already marks real boundaries, so the loader chunks on those and every chunk is one complete provision.

**Citation baked into the embedded text.** Each chunk reads `Section 9(1): A person is guilty of burglary if...`. The identity is embedded with the content, so a subsection retrieved alone still reads as a citation. Metadata alone would not do this, since metadata is never embedded and the model only sees the text.

**No overlap.** Overlap exists to rescue ideas cut at arbitrary boundaries. There are no arbitrary boundaries here, so it buys nothing.

**IDs derived, not read.** Section number plus subsection number is not unique, because Schedules repeat the same numbering as the body. The chunk ID is an md5 of the provision's URI, built from the parent element rather than read off the child, after finding a published Act where one child URI pointed at the wrong section.

**Whole-Act fetch.** One request returns every section, against dozens if fetching section by section and guessing where to stop.

**Embeddings local, generation swappable.** Embedding runs on Ollama, so re-indexing is free and the corpus never leaves the machine. Generation goes through `init_chat_model`, making the provider configuration rather than code.

## Retrieval quality

Ten questions, each paired with the section that should answer it, scored on whether that section appears in the top k.

**hit@4: 10/10.** Questions are phrased the way a member of the public would ask ("is hacking into a computer illegal", "is taking someone's car without permission a crime") rather than in statutory language, so the score reflects retrieval bridging plain English to legal drafting. An earlier corpus scored 9/10, with the failure traced to near-identical Schedule paragraphs crowding out the section that answered the question.

## Tests

Four assertions run against the loaded chunks before anything is indexed:

| Check | Catches |
|---|---|
| Non-empty result | Dead URL, wrong namespace |
| Every chunk has a URI | Provisions the parser could not identify |
| URIs unique | Schedule and section numbering collisions |
| Every chunk has text | Provisions whose text the parser could not reach |

Catching duplicates here rather than in the database gives a readable failure instead of a dump of hashes. A grounding check confirms the model refuses questions the corpus cannot answer.

## Working with the source data

The published XML is well structured but not clean. Five issues surfaced during development and each is handled in the loader: the API serves XML rather than JSON and must be parsed from raw bytes; every tag sits in an unprefixed namespace that silently returns nothing if omitted; section numbers can hide behind nested commentary tags; text quoted from other Acts appears inline and has to be excluded or it gets indexed as the wrong Act's law; and amended sections store their text in markup the obvious element search never reaches. Each was found by running the pipeline and reading the failure, not from documentation.

## Running

Ollama must be running first. Then:

```python
store = index()                                       # fetch, chunk, embed, persist
evaluate(store)                                       # hit@4 across the eval set
print(ask(store, "what counts as theft?", debug=True))
```

`debug=True` prints the retrieved sections alongside the answer so grounding can be checked by eye.

### Optional: hosted model

Generation defaults to local `llama3.2`. Two constants switch it to a hosted provider with no other code change:

```python
LLM_MODEL, LLM_PROVIDER = "claude-sonnet-4-6", "anthropic"
```

The matching API key goes in `.env`. Embeddings stay local either way, so only the question and the four retrieved chunks are sent to the provider.

## Known gaps

- Sections without numbered subsections are skipped.
- The eval set is small and every question currently passes, which suggests it needs harder cases rather than that retrieval is perfect.
- No amendment tracking, so the corpus is a snapshot rather than current law.
- Vector search only. Exact-term lookups ("what does section 9 say") would benefit from hybrid keyword search.

## Setup

### Windows (PowerShell)

```powershell
irm https://astral.sh/uv/install.ps1 | iex

uv venv --python 3.12
.venv\Scripts\Activate.ps1

uv pip install langchain langchain-core langchain-chroma langchain-ollama `
               langchain-anthropic requests python-dotenv ipykernel

winget install Ollama.Ollama
ollama pull nomic-embed-text
ollama pull llama3.2
```

If `Activate.ps1` is blocked: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, once, then retry. Reopen the terminal after installing uv or Ollama so PATH updates.

### macOS / Linux

```bash
brew install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install langchain langchain-core langchain-chroma langchain-ollama \
               langchain-anthropic requests python-dotenv ipykernel
ollama pull nomic-embed-text
ollama pull llama3.2
```

Select `.venv` as the kernel in VS Code (top right).

**Why Python 3.12.** Compiled packages (Chroma, torch) ship prebuilt wheels for new Python versions months late. On the newest interpreter pip falls back to compiling from source, which is slow and fails often.

## Credits

Structure follows LangChain's [RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) series. The loader, chunking strategy, and legislation-specific handling are original work against the [legislation.gov.uk API](https://www.legislation.gov.uk/developer).