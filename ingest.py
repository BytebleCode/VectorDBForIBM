"""
ChromaDB Text File Ingestion Pipeline

Reads .txt files from an input directory, chunks them, and stores
vectorized embeddings in a persistent ChromaDB database.
"""

import argparse
import os
import sys
import hashlib
from pathlib import Path

import chromadb


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def make_doc_id(filename: str, chunk_index: int) -> str:
    """Create a deterministic document ID from filename and chunk index."""
    raw = f"{filename}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def read_txt_files(input_dir: str, encoding: str) -> list[tuple[str, str]]:
    """Read all .txt files from the input directory. Returns list of (filename, content)."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a valid directory.")
        sys.exit(1)

    files = sorted(input_path.glob("*.txt"))
    if not files:
        print(f"No .txt files found in '{input_dir}'.")
        sys.exit(1)

    results = []
    for fpath in files:
        try:
            content = fpath.read_text(encoding=encoding)
            if content.strip():
                results.append((fpath.name, content))
            else:
                print(f"  Skipping empty file: {fpath.name}")
        except UnicodeDecodeError as e:
            print(f"  Skipping {fpath.name} (encoding error: {e})")
        except Exception as e:
            print(f"  Skipping {fpath.name} (read error: {e})")

    if not results:
        print("No readable .txt files with content found.")
        sys.exit(1)

    return results


def ingest(
    input_dir: str,
    db_dir: str,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding: str,
    batch_size: int,
    reset: bool,
):
    """Main ingestion pipeline."""
    # 1. Read files
    print(f"Reading .txt files from: {input_dir}")
    file_contents = read_txt_files(input_dir, encoding)
    print(f"  Found {len(file_contents)} file(s).\n")

    # 2. Connect to ChromaDB
    print(f"Opening ChromaDB at: {db_dir}")
    client = chromadb.PersistentClient(path=db_dir)

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"  Deleted existing collection '{collection_name}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Collection '{collection_name}' ready (existing docs: {collection.count()}).\n")

    # 3. Chunk and ingest
    total_chunks = 0
    for filename, content in file_contents:
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        print(f"  {filename}: {len(content)} chars -> {len(chunks)} chunk(s)")

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = make_doc_id(filename, i)
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "source_file": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk),
            })

        # Upsert in batches
        for b_start in range(0, len(ids), batch_size):
            b_end = b_start + batch_size
            collection.upsert(
                ids=ids[b_start:b_end],
                documents=documents[b_start:b_end],
                metadatas=metadatas[b_start:b_end],
            )

        total_chunks += len(chunks)

    print(f"\nDone. Ingested {total_chunks} chunks from {len(file_contents)} file(s).")
    print(f"Collection now has {collection.count()} documents total.")


def query_db(db_dir: str, collection_name: str, query_text: str, n_results: int):
    """Query the database and print results."""
    client = chromadb.PersistentClient(path=db_dir)
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        print(f"Collection '{collection_name}' not found. Run ingestion first.")
        sys.exit(1)

    results = collection.query(query_texts=[query_text], n_results=n_results)

    print(f"Query: \"{query_text}\"")
    print(f"Top {n_results} results:\n")

    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"--- Result {i + 1} (distance: {dist:.4f}) ---")
        print(f"Source: {meta['source_file']} | Chunk {meta['chunk_index'] + 1}/{meta['total_chunks']}")
        print(doc[:300])
        if len(doc) > 300:
            print("...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="ChromaDB Text Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py ingest
  python ingest.py ingest --input-dir ./my_texts --chunk-size 500
  python ingest.py ingest --reset
  python ingest.py query "What is machine learning?"
  python ingest.py query "search term" --n-results 10
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest .txt files into ChromaDB")
    ingest_parser.add_argument("--input-dir", default="./input_texts", help="Directory containing .txt files (default: ./input_texts)")
    ingest_parser.add_argument("--db-dir", default="./chroma_db", help="ChromaDB storage directory (default: ./chroma_db)")
    ingest_parser.add_argument("--collection", default="documents", help="Collection name (default: documents)")
    ingest_parser.add_argument("--chunk-size", type=int, default=1000, help="Characters per chunk (default: 1000)")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks (default: 200)")
    ingest_parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    ingest_parser.add_argument("--batch-size", type=int, default=100, help="ChromaDB upsert batch size (default: 100)")
    ingest_parser.add_argument("--reset", action="store_true", help="Delete existing collection before ingesting")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query the ChromaDB collection")
    query_parser.add_argument("query_text", help="Text to search for")
    query_parser.add_argument("--db-dir", default="./chroma_db", help="ChromaDB storage directory (default: ./chroma_db)")
    query_parser.add_argument("--collection", default="documents", help="Collection name (default: documents)")
    query_parser.add_argument("--n-results", type=int, default=5, help="Number of results (default: 5)")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(
            input_dir=args.input_dir,
            db_dir=args.db_dir,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            encoding=args.encoding,
            batch_size=args.batch_size,
            reset=args.reset,
        )
    elif args.command == "query":
        query_db(
            db_dir=args.db_dir,
            collection_name=args.collection,
            query_text=args.query_text,
            n_results=args.n_results,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
