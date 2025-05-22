import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
import numpy as np
import os
from datetime import datetime
from content_loader import Page
from text_chunker import Chunk, TextChunker, ChunkingConfig


class ChromaConfig(BaseModel):
    """Configuration for ChromaDB"""
    persist_directory: str = "./chroma_db"
    collection_name: str = "documents"
    embedding_function_name: str = "openai"  # supported: openai, cohere, default


class QueryResult(BaseModel):
    """Result of a vector search query"""
    id: str
    score: float
    document: Dict[str, Any]
    metadata: Dict[str, Any]


class VectorStore:
    """Vector store for document similarity search using ChromaDB"""
    
    def __init__(self, config: ChromaConfig = ChromaConfig(), chunking_config: Optional[ChunkingConfig] = None):
        """
        Initialize the vector store
        
        Args:
            config: Configuration for ChromaDB
            chunking_config: Optional configuration for text chunking. If provided, enables chunking.
        """
        self.config = config
        self.chunking_enabled = chunking_config is not None
        self.chunker = TextChunker(chunking_config) if chunking_config else None
        
        os.makedirs(config.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=config.persist_directory)
        
        # Set up embedding function
        if config.embedding_function_name == "openai":
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
        elif config.embedding_function_name == "cohere":
            self.embedding_function = embedding_functions.CohereEmbeddingFunction(
                api_key=os.environ.get("COHERE_API_KEY"),
                model_name="embed-english-v3.0"
            )
        else:
            # Default embedding function (all-MiniLM-L6-v2)
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
    
    def add_pages(self, pages: List[Page], batch_size: int = 100) -> None:
        """
        Add multiple pages to the vector store
        
        Args:
            pages: List of Page objects to add
            batch_size: Number of documents to add in a single batch
        """
        if self.chunking_enabled and self.chunker:
            # Chunk pages and add chunks
            chunks = self.chunker.chunk_pages(pages)
            self.add_chunks(chunks, batch_size)
            print(f"Added {len(pages)} pages as {len(chunks)} chunks to the vector store")
        else:
            # Add whole pages without chunking
            for i in range(0, len(pages), batch_size):
                batch = pages[i:i+batch_size]
                
                ids = [page.id for page in batch]
                documents = [page.content if page.content else "" for page in batch]
                
                # Process metadata to remove None values - ChromaDB doesn't accept None values
                metadatas = []
                for page in batch:
                    # Get metadata as dict and filter out None values
                    metadata = page.meta.model_dump()
                    filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
                    
                    # Ensure all values are primitive types accepted by ChromaDB
                    for key, value in list(filtered_metadata.items()):
                        # Convert lists and dicts to strings
                        if isinstance(value, (list, dict)):
                            filtered_metadata[key] = str(value)
                        # Remove other non-primitive types
                        elif not isinstance(value, (str, int, float, bool)):
                            filtered_metadata.pop(key)
                    
                    metadatas.append(filtered_metadata)
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                
            print(f"Added {len(pages)} pages to the vector store")
    
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> None:
        """
        Add multiple chunks to the vector store
        
        Args:
            chunks: List of Chunk objects to add
            batch_size: Number of chunks to add in a single batch
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            ids = [chunk.id for chunk in batch]
            documents = [chunk.text for chunk in batch]
            
            # Process metadata to remove None values - ChromaDB doesn't accept None values
            metadatas = []
            for chunk in batch:
                # Get metadata as dict and filter out None values
                metadata = chunk.metadata.model_dump()
                filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
                
                # Ensure all values are primitive types accepted by ChromaDB
                for key, value in list(filtered_metadata.items()):
                    # Convert lists and dicts to strings
                    if isinstance(value, (list, dict)):
                        filtered_metadata[key] = str(value)
                    # Remove other non-primitive types
                    elif not isinstance(value, (str, int, float, bool)):
                        filtered_metadata.pop(key)
                
                metadatas.append(filtered_metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        include_documents: bool = True
    ) -> List[QueryResult]:
        """
        Query the vector store for similar documents
        
        Args:
            query_text: The text to search for
            n_results: Number of results to return
            where_filter: Filter to apply to the search (e.g. {"meta_author": "John"})
            include_documents: Whether to include the document content in the results
            
        Returns:
            List of QueryResult objects sorted by relevance
        """
        include = ["metadatas", "distances", "documents"] if include_documents else ["metadatas", "distances"]
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
            include=include
        )
        
        query_results = []
        
        # Process results
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]
            
            # Convert distance to similarity score (1 = exact match, 0 = completely different)
            # Since we're using cosine distance, we need to convert it to similarity
            similarity_score = 1 - distance
            
            result = QueryResult(
                id=doc_id,
                score=similarity_score,
                metadata=metadata,
                document={"content": results["documents"][0][i]} if include_documents else {}
            )
            
            query_results.append(result)
            
        return query_results
    
    def update_page(self, page: Page) -> None:
        """
        Update a page in the vector store
        
        Args:
            page: The Page object to update
        """
        # Delete the existing entry if it exists
        try:
            self.collection.delete(ids=[page.id])
        except Exception:
            pass  # Document might not exist
        
        # Process metadata to remove None values - ChromaDB doesn't accept None values
        metadata = page.meta.model_dump()
        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Ensure all values are primitive types accepted by ChromaDB
        for key, value in list(filtered_metadata.items()):
            # Convert lists and dicts to strings
            if isinstance(value, (list, dict)):
                filtered_metadata[key] = str(value)
            # Remove other non-primitive types
            elif not isinstance(value, (str, int, float, bool)):
                filtered_metadata.pop(key)
            
        # Add the updated page
        self.collection.add(
            ids=[page.id],
            documents=[page.content if page.content else ""],
            metadatas=[filtered_metadata]
        )
    
    def delete_page(self, page_id: str) -> None:
        """
        Delete a page from the vector store
        
        Args:
            page_id: The ID of the page to delete
        """
        self.collection.delete(ids=[page_id])
    
    def get_page_count(self) -> int:
        """Get the total number of pages in the vector store"""
        return self.collection.count()
    
    def list_all_ids(self) -> List[str]:
        """Get a list of all page IDs in the vector store"""
        # Efficiently only retrieve IDs
        results = self.collection.get(include=["documents"])
        return results["ids"]
    
    def clear_collection(self) -> None:
        """Delete all documents in the collection"""
        self.collection.delete()
        
    def get_by_id(self, page_id: str) -> Optional[QueryResult]:
        """
        Get a page by its ID
        
        Args:
            page_id: The ID of the page to retrieve
            
        Returns:
            QueryResult object or None if not found
        """
        try:
            result = self.collection.get(
                ids=[page_id],
                include=["metadatas", "documents"]
            )
            
            if not result["ids"]:
                return None
                
            return QueryResult(
                id=result["ids"][0],
                score=1.0,  # Perfect score for exact ID match
                metadata=result["metadatas"][0],
                document={"content": result["documents"][0]}
            )
        except Exception:
            return None
            
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        return {
            "count": self.collection.count(),
            "name": self.config.collection_name,
            "embedding_function": self.config.embedding_function_name,
            "persist_directory": self.config.persist_directory
        }


# Usage example
if __name__ == "__main__":
    # Setup vector store with chunking
    chroma_config = ChromaConfig(
        persist_directory="./chroma_db",
        collection_name="documents",
        embedding_function_name="default"  # No API key needed
    )
    
    chunking_config = ChunkingConfig(
        chunk_size=500,
        chunk_overlap=100,
        respect_paragraph=True
    )
    
    # Create vector store with chunking enabled
    vector_store = VectorStore(chroma_config, chunking_config)
    
    # Example usage with an in-memory page
    from content_loader import PageMeta, Page
    
    # Create a sample page with longer content to demonstrate chunking
    meta = PageMeta(
        url="https://example.com/sample",
        title="Sample Document about Vector Search",
        author="John Doe",
        pub_time="2023-01-01T00:00:00Z"
    )
    
    # Create longer content that will be split into multiple chunks
    content = """
    Vector search is a technique used to find similar items in a large dataset.
    It works by converting items into high-dimensional vectors and finding the nearest neighbors in vector space.
    
    Vector databases like ChromaDB, Pinecone, and Weaviate are specialized systems designed for efficient vector similarity search.
    They use algorithms such as approximate nearest neighbor search to quickly find similar vectors.
    
    One common application of vector search is semantic search, which goes beyond keyword matching to understand
    the meaning and context of a query. This is often powered by embedding models that convert text into vectors.
    
    Chunking is an important technique when working with long documents in vector search.
    Long documents are split into smaller, more manageable chunks with some overlap between them.
    This allows for more precise retrieval of relevant information.
    
    The chunking process typically involves:
    1. Splitting text by size or semantic boundaries
    2. Including overlap between chunks to preserve context
    3. Storing metadata with each chunk to track its source
    
    When a query is made, the system can retrieve the most relevant chunks rather than entire documents,
    making the results more focused and accurate.
    """
    
    page = Page(meta=meta, content=content)
    
    # Add the page to the vector store (it will be automatically chunked)
    vector_store.add_pages([page])
    
    # Query the vector store - note how it returns specific chunks rather than whole documents
    print("\n--- Basic Query ---")
    results = vector_store.query("chunking for vector search", n_results=2)
    
    # Print results
    for result in results:
        print(f"ID: {result.id}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.metadata.get('source_url')}")
        print(f"Chunk: {result.metadata.get('chunk_index')+1} of {result.metadata.get('total_chunks')}")
        print(f"Content: {result.document.get('content')[:150]}...")
        print("-" * 50)
        
    # Query with specific metadata filter
    print("\n--- Filtered Query ---")
    results = vector_store.query(
        "vector database", 
        n_results=2,
        where_filter={"author": "John Doe"}
    )
    
    # Print results
    for result in results:
        print(f"ID: {result.id}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.metadata.get('source_url')}")
        if 'chunk_index' in result.metadata:
            print(f"Chunk: {result.metadata.get('chunk_index')+1} of {result.metadata.get('total_chunks')}")
        print(f"Content: {result.document.get('content')[:150]}...")
        print("-" * 50)