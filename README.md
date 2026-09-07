# Legislation RAG Pipeline

A retrieval augmented generation pipeline over UK primary legislation. Acts are fetched from the legislation.gov.uk API as XML, split into chunks along the law's own structural boundaries, embedded locally through Ollama, stored in Chroma, then retrieved at query time and passed to an LLM as grounded context.

Current corpus: Commons Act 2006 (`ukpga/2006/26`), 382 subsections. Any Act on legislation.gov.uk works by changing one URL.

## Architecture

```
legislation.gov.uk API -> XML parse -> subsection chunks -> embed (Ollama)
  -> Chroma -> retrieve (k=4) -> generate
```

Six cells, one job each: config, load, store, index, retrieve, generate.

## Design decisions

**Chunk by subsection, not by character count.** The single most important choice here. `RecursiveCharacterTextSplitter` cuts every N characters wherever that happens to fall, which for legal text means slicing mid-clause: one chunk gets half a condition, the next gets the rest, and both embed as blurry, unmatchable fragments. Legislation already carries its own boundaries in the XML (`P1` = section, `P2` = numbered subsection, `P3` = lettered sub-point), so the loader chunks on those. Every chunk is one complete legal provision. There is no character-splitting step in this pipeline at all.

**No `split()` function.** Chunking happens inside `load()` because the source format dictates the boundaries; chunk size is a property of the data here, not a tuning parameter. The tradeoff: a function named `load` owns a decision you might expect to live elsewhere. The alternative (load whole sections, then re-split them) keeps function roles symmetrical but takes text apart that was just assembled. Chose the honest version over the tidy one.

**No overlap.** `chunk_overlap` exists to rescue ideas cut at arbitrary boundaries. There are no arbitrary boundaries here, so nothing needs rescuing. What replaces it is the citation prefix below.

**Citation baked into the embedded text.** Each chunk's `page_content` starts with `Section 9(2): ` before the legal text. That identity is embedded along with the content, so a subsection retrieved on its own still reads as a complete citation rather than an orphaned sentence with no idea what Act or section it came from. Metadata alone wouldn't do this, since metadata isn't embedded and the LLM only ever sees `page_content`.

**Whole-Act fetch, not per-section.** `/ukpga/2006/26/data.xml` returns every section in one request; the alternative loops `/section/1`, `/section/2`, ... and has to guess where to stop and handle gaps. The loop over `P1` elements handles the whole Act unchanged. Rate limit is 3,000 requests per 5 minutes, so neither approach strains it, but one request beats fifty.

**IDs from `DocumentURI`, not section numbers.** Section number plus subsection number is *not* unique across an Act: Schedules carry their own paragraph numbering that collides with body sections (44 collisions in the Commons Act alone). Every genuine provision carries a `DocumentURI` attribute which is canonical and globally unique, so the chunk ID is just an md5 of that. Re-running the indexer upserts instead of duplicating. `index(rebuild=True)` wipes and re-embeds.

**Retrieval depth `k=4`.** A dial, not a score: too low risks missing a relevant provision, too high pulls in weakly related sections that burn prompt budget and bury the good ones.

**Embeddings local, generation swappable.** `nomic-embed-text` through Ollama, always local, so re-indexing costs nothing and never ships the corpus anywhere. Generation goes through `init_chat_model`, so the provider is configuration rather than code. Local `llama3.2` is free and private; a hosted model answers better. Privacy scope, precisely: indexing is always local, but a hosted generator receives the question plus the four retrieved chunks on every query.

**Why Python 3.12.** Compiled packages (Chroma, torch) ship prebuilt wheels for new Python versions months late. On the newest interpreter pip falls back to compiling from source, which is slow and fails often.

## Source data gotchas

Five things the legislation XML does that break naive parsing, each found by running the code:

1. **It's XML, not JSON.** The `requests` response is `application/xml`. Parse `req.content` (raw bytes), never `req.text` — the document carries its own encoding declaration and a strict parser rejects a pre-decoded string.
2. **Everything sits in one unprefixed default namespace.** Every tag must be searched as `{http://www.legislation.gov.uk/namespaces/legislation}P1`. Search without it and ElementTree returns nothing — no error, just silence.
3. **`Pnumber` can contain nested `CommentaryRef` tags before the digit**, so `.text` returns `None`. Use `itertext()`.
4. **Quoted text from other Acts appears inline.** Section 50 amends the Commons Act 1899 and quotes the inserted provisions, which are `P1`/`P2` elements too. They lack a `DocumentURI`, and that absence is the only reliable marker — the loader skips them. Without this they'd be indexed as if they were this Act's own law, which is a correctness bug, not just a duplicate-ID one.
5. **Amended sections (e.g. `15A`) hold their text inside markup that `Text` elements don't reach.** Text is collected from `P2para` via `.iter()` across `.text` and `.tail`, skipping `Pnumber` so sub-point letters don't bleed into the prose.

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

## Running

**Ollama must be running before anything else.** On Windows it's a desktop app: launch it from the Start menu and it sits in the system tray serving on localhost. Embeddings and local generation both fail with connection errors or 404s if it isn't up, or if the models aren't pulled (`ollama list` to check).

Then run the notebook cells in order. Cell 4 (`index()`) embeds 382 subsections locally and takes a minute or two on first run; subsequent runs upsert and are fast.

```python
docs = load(ACT_URL)        # 382 Documents, one per subsection
store = index()             # embed + persist to ./chroma_db
show(retrieve(store, "when can a right of common be severed from land"))
print(ask(store, "Can a right of common be severed from the land it is attached to?"))
```

### Switching to a hosted model

Two constants in cell 1, then re-run cell 6 so `llm` is rebuilt:

```python
LLM_MODEL, LLM_PROVIDER = "claude-sonnet-4-6", "anthropic"
```

`.env` needs `ANTHROPIC_API_KEY=...`. API credits are billed separately from any Claude subscription; a Pro plan does not fund API calls. Set a spending limit in the console before the key goes in the file.

## Tests

Run after any loader change or when pointing at a new Act:

```python
docs = load(ACT_URL)
assert docs, "loader returned nothing"
assert all(d.metadata["uri"] for d in docs), "missing URI"
assert len({d.metadata["uri"] for d in docs}) == len(docs), "duplicate URIs"
assert all(len(d.page_content.split(": ", 1)[1]) > 0 for d in docs), "empty content"
print(f"{len(docs)} docs, all valid")
```

These four catch every failure mode hit during development: dead URL or broken namespace, missing URIs, the Schedule/section number collision, and provisions whose text the parser couldn't reach. Catching duplicates here rather than in Chroma gives a readable message instead of a dump of md5 hashes.

Grounding check, which should refuse rather than answer:

```python
print(ask(store, "What is the capital of France?"))
```

## Known gaps

- **Sections without subsections are skipped.** The inner loop needs at least one `P2`; an unsubdivided section produces nothing.
- **Schedule paragraphs are cited as "Section N".** They're indexed correctly and uniquely, but the citation prefix mislabels them.
- **Single Act.** `ACT_URL` is one constant; a corpus needs a list and a loop. URI-based IDs already handle cross-Act uniqueness.
- **No retrieval eval.** Retrieval looks right by inspection but isn't measured. Next step is ten questions with expected sections, scoring how often the right chunk lands in the top k.

## Credits

Structure follows LangChain's [RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) series. The loader, chunking strategy, and legislation-specific handling are original work against the [legislation.gov.uk API](https://www.legislation.gov.uk/developer).