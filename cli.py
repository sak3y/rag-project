"""Terminal chat over your docs.

  python cli.py                  interactive chat (default)
  python cli.py "who am i"       one-shot answer
  python cli.py --index          rebuild the index then chat
  python cli.py --show           print retrieved chunks before each answer

In chat:  /show  toggle chunk display   /index  reindex   /k 6  change k   /exit
"""


import argparse, sys
import rag


def print_hits(hits):
    for doc, score in hits:
        print(f"  [{score:.3f}] {doc.metadata['source']}: "
              f"{doc.page_content[:120].replace(chr(10), ' ')}")
    print()


def answer(store, llm, q, k, show):
    gen = rag.stream(store, q, k, llm)
    try:
        while True:
            print(next(gen), end="", flush=True)
    except StopIteration as done:
        print("\n")
        if show:
            print_hits(done.value)


def chat(store, llm, k, show):
    print(f"rag chat  |  {rag.LLM_MODEL} via {rag.LLM_PROVIDER}  |  /exit to quit\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not q:
            continue
        if q in ("/exit", "/quit", "exit", "quit"):
            return
        if q == "/show":
            show = not show; print(f"show chunks: {show}\n"); continue
        if q == "/index":
            store, n, f = rag.index(rebuild=True); print(f"{n} chunks from {f} files\n"); continue
        if q.startswith("/k "):
            k = int(q.split()[1]); print(f"k = {k}\n"); continue
        print("rag > ", end="", flush=True)
        answer(store, llm, q, k, show)


def main():
    p = argparse.ArgumentParser(description="chat with your docs")
    p.add_argument("question", nargs="?", help="one-shot question; omit for chat mode")
    p.add_argument("--index", action="store_true", help="rebuild the index first")
    p.add_argument("--show", action="store_true", help="print retrieved chunks")
    p.add_argument("-k", type=int, default=4)
    a = p.parse_args()

    if a.index:
        _, n, f = rag.index(rebuild=True)
        print(f"indexed {n} chunks from {f} files\n")
    store = rag.get_store()
    if store._collection.count() == 0:
        _, n, f = rag.index()
        print(f"indexed {n} chunks from {f} files\n")
    llm = rag.get_llm()

    if a.question:
        answer(store, llm, a.question, a.k, a.show)
    else:
        chat(store, llm, a.k, a.show)


if __name__ == "__main__":
    sys.exit(main())