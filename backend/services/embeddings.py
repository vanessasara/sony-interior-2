"""
Embeddings service for generating text embeddings and performing vector similarity search.
Phase 11: Embeddings and RAG Implementation
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pgvector.asyncpg import Vector
import asyncpg
from functools import lru_cache

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Embedding model - using MiniLM for efficiency (384 dimensions)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 300  # words
CHUNK_OVERLAP = 50  # words


class EmbeddingsService:
    """Service for generating embeddings and performing vector similarity search."""

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None

    async def initialize(self):
        """Initialize the embedding model."""
        if self.model is None:
            # Run model loading in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(EMBEDDING_MODEL_NAME)
            )
            print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Input text string

        Returns:
            List of floats representing the embedding vector
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if not text or not text.strip():
            return [0.0] * 384  # Return zero vector for empty text

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts.

        Args:
            texts: List of input text strings

        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Filter out empty texts
        valid_texts = [t if t and t.strip() else " " for t in texts]

        # Generate embeddings in batch
        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


# Global service instance
_embeddings_service: Optional[EmbeddingsService] = None


async def get_embeddings_service() -> EmbeddingsService:
    """Get or create the embeddings service instance."""
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = EmbeddingsService()
        await _embeddings_service.initialize()
    return _embeddings_service


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        chunk_size: Maximum words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # Split into words
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap

    return chunks


def chunk_product_for_embedding(product: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create embedding-ready chunks from product data.

    Args:
        product: Product dictionary from Sanity

    Returns:
        List of chunks with text and metadata
    """
    chunks = []

    # Combine relevant product fields into embedding text
    name = product.get("name", "")
    description = product.get("description", "")
    short_description = product.get("shortDescription", "")
    category = product.get("category", {}).get("name", "") if isinstance(product.get("category"), dict) else str(product.get("category", ""))
    materials = ", ".join(product.get("materials", [])) if product.get("materials") else ""
    dimensions = product.get("dimensions", {})
    colors = ", ".join([c.get("name", "") for c in product.get("colors", [])]) if product.get("colors") else ""

    # Create main product chunk
    main_text_parts = [name]
    if short_description:
        main_text_parts.append(short_description)
    if description:
        main_text_parts.append(str(description))
    if category:
        main_text_parts.append(f"Category: {category}")
    if materials:
        main_text_parts.append(f"Materials: {materials}")
    if colors:
        main_text_parts.append(f"Colors: {colors}")
    if dimensions:
        dim_str = f"Dimensions: {dimensions.get('width', 'N/A')}W x {dimensions.get('height', 'N/A')}H x {dimensions.get('depth', 'N/A')}D"
        main_text_parts.append(dim_str)

    main_text = ". ".join(main_text_parts)

    if main_text.strip():
        chunks.append({
            "text": main_text,
            "source_type": "product",
            "source_id": str(product.get("_id", "")),
            "source_name": name,
            "metadata": {
                "category": category,
                "price": product.get("price"),
                "stock_status": product.get("stockStatus")
            }
        })

    return chunks


def chunk_page_content_for_embedding(page_slug: str, content: str, title: str = "") -> List[Dict[str, Any]]:
    """
    Create embedding-ready chunks from page content.

    Args:
        page_slug: URL-friendly page identifier
        content: Page content text
        title: Page title

    Returns:
        List of chunks with text and metadata
    """
    chunks = []
    text_chunks = chunk_text(content)

    for i, chunk in enumerate(text_chunks):
        chunks.append({
            "text": chunk,
            "source_type": "page_content",
            "source_id": page_slug,
            "source_name": title or page_slug,
            "metadata": {
                "chunk_index": i,
                "page": page_slug
            }
        })

    return chunks


async def get_db_pool():
    """Get database connection pool."""
    return await asyncpg.create_pool(DATABASE_URL)


async def store_embeddings(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> bool:
    """
    Store embeddings in the database.

    Args:
        chunks: List of chunk dictionaries with text and metadata
        embeddings: List of embedding vectors

    Returns:
        True if successful, False otherwise
    """
    if not chunks or not embeddings:
        return False

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Insert embeddings
            for chunk, embedding in zip(chunks, embeddings):
                await conn.execute(
                    """
                    INSERT INTO document_embeddings (embedding_id, source_type, source_id, content_chunk, embedding, metadata)
                    VALUES (gen_random_uuid(), $1, $2, $3, $4, $5)
                    ON CONFLICT (embedding_id) DO NOTHING
                    """,
                    chunk["source_type"],
                    chunk["source_id"],
                    chunk["text"],
                    Vector(embedding),
                    chunk.get("metadata", {})
                )

        await pool.close()
        return True
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return False


async def search_similar_content(
    query: str,
    source_type: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search for relevant content.

    Args:
        query: Search query text
        source_type: Optional filter by source type (e.g., 'product', 'page_content')
        limit: Maximum number of results

    Returns:
        List of relevant content chunks with similarity scores
    """
    try:
        # Generate query embedding
        service = await get_embeddings_service()
        query_embedding = service.generate_embedding(query)

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Build query with optional source_type filter
            if source_type:
                results = await conn.fetch(
                    """
                    SELECT
                        source_type,
                        source_id,
                        content_chunk,
                        metadata,
                        1 - (embedding <=> $1) as similarity
                    FROM document_embeddings
                    WHERE source_type = $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                    """,
                    Vector(query_embedding),
                    source_type,
                    limit
                )
            else:
                results = await conn.fetch(
                    """
                    SELECT
                        source_type,
                        source_id,
                        content_chunk,
                        metadata,
                        1 - (embedding <=> $1) as similarity
                    FROM document_embeddings
                    ORDER BY embedding <=> $1
                    LIMIT $2
                    """,
                    Vector(query_embedding),
                    limit
                )

        await pool.close()

        # Format results
        return [
            {
                "source_type": r["source_type"],
                "source_id": r["source_id"],
                "content": r["content_chunk"],
                "similarity": float(r["similarity"]),
                "metadata": r["metadata"]
            }
            for r in results
        ]

    except Exception as e:
        print(f"Error searching similar content: {e}")
        return []


def build_rag_context(query: str, results: List[Dict[str, Any]], max_length: int = 2000) -> str:
    """
    Build context string from search results for LLM consumption.

    Args:
        query: Original user query
        results: Search results from vector similarity
        max_length: Maximum character length of context

    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant information found."

    context_parts = ["Relevant information:"]
    current_length = len("\n".join(context_parts))

    for i, result in enumerate(results):
        content = result["content"]
        source = result.get("source_id", "unknown")
        source_type = result.get("source_type", "unknown")
        similarity = result.get("similarity", 0)

        # Add source attribution
        source_info = f"[Source: {source_type} - {source}, Relevance: {similarity:.2f}]"

        # Check if adding this would exceed max length
        chunk = f"\n\n--- Result {i + 1} ---\n{content}\n{source_info}"
        if current_length + len(chunk) > max_length:
            break

        context_parts.append(chunk)
        current_length += len(chunk)

    return "\n".join(context_parts)


# Standalone functions for direct use

async def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text."""
    service = await get_embeddings_service()
    return service.generate_embedding(text)


async def embed_products(products: List[Dict[str, Any]]) -> bool:
    """
    Embed all products and store in database.

    Args:
        products: List of product dictionaries from Sanity

    Returns:
        True if successful
    """
    service = await get_embeddings_service()

    all_chunks = []
    for product in products:
        chunks = chunk_product_for_embedding(product)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks to embed")
        return False

    # Generate embeddings in batch
    texts = [c["text"] for c in all_chunks]
    embeddings = service.generate_embeddings_batch(texts)

    # Store in database
    return await store_embeddings(all_chunks, embeddings)


async def embed_page_content(
    page_slug: str,
    content: str,
    title: str = ""
) -> bool:
    """
    Embed page content and store in database.

    Args:
        page_slug: URL-friendly page identifier
        content: Page content text
        title: Page title

    Returns:
        True if successful
    """
    service = await get_embeddings_service()

    chunks = chunk_page_content_for_embedding(page_slug, content, title)

    if not chunks:
        print("No content to embed")
        return False

    texts = [c["text"] for c in chunks]
    embeddings = service.generate_embeddings_batch(texts)

    return await store_embeddings(chunks, embeddings)


async def rag_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Perform RAG search - generate embedding, search, and build context.

    Args:
        query: User query
        limit: Number of results to retrieve

    Returns:
        Dictionary with results and formatted context
    """
    results = await search_similar_content(query, limit=limit)
    context = build_rag_context(query, results)

    return {
        "results": results,
        "context": context,
        "query": query
    }
