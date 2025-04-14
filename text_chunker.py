from typing import List, Dict, Any, Optional, Union, Callable, Iterator, Tuple
from pydantic import BaseModel, Field, computed_field
import re
import hashlib
from content_loader import Page, PageMeta


class ChunkingConfig(BaseModel):
    """Configuration for text chunking strategies"""
    chunk_size: int = 1000  # Target size of each chunk in characters
    chunk_overlap: int = 200  # Number of characters to overlap between chunks
    separator: str = " "  # Default separator for chunk boundaries
    respect_paragraph: bool = True  # Try to respect paragraph boundaries
    include_metadata: bool = True  # Include original metadata in each chunk
    append_chunk_info: bool = True  # Add chunk number info to metadata


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk"""
    source_id: str  # ID of the source document
    source_url: str  # URL of the source document
    chunk_index: int  # Index of this chunk in the source document
    total_chunks: int  # Total number of chunks in the source document
    start_char: int  # Start character position in the source text
    end_char: int  # End character position in the source text
    
    # Allow additional fields beyond the predefined ones
    model_config = {
        "extra": "allow"
    }
    
    @computed_field
    def id(self) -> str:
        """Generate a unique ID for the chunk based on source ID and chunk index"""
        chunk_id_str = f"{self.source_id}_{self.chunk_index}"
        return hashlib.md5(chunk_id_str.encode()).hexdigest()


class Chunk(BaseModel):
    """A chunk of text with metadata"""
    text: str
    metadata: ChunkMetadata
    
    @computed_field
    def id(self) -> str:
        """Forward the ID from metadata for easier access"""
        return self.metadata.id


class TextChunker:
    """Text chunker with various chunking strategies and overlap control"""
    
    def __init__(self, config: ChunkingConfig = ChunkingConfig()):
        """Initialize the text chunker with the given configuration"""
        self.config = config
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Split text into chunks according to the chunking configuration
        
        Args:
            text: The text to chunk
            metadata: Metadata to include with each chunk
            
        Returns:
            List of Chunk objects
        """
        if not text or text.strip() == "":
            return []
            
        # Determine the chunking strategy based on configuration
        if self.config.respect_paragraph:
            chunks = self._chunk_by_paragraph(text)
        else:
            chunks = self._chunk_by_size(text)
            
        # Create Chunk objects with metadata
        result = []
        source_id = metadata.get("id", metadata.get("url", "unknown"))
        total_chunks = len(chunks)
        
        for i, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            # Create chunk metadata
            chunk_metadata = {
                "source_id": source_id,
                "source_url": metadata.get("url", ""),
                "chunk_index": i,
                "total_chunks": total_chunks,
                "start_char": start_pos,
                "end_char": end_pos,
            }
            
            # Add original metadata if configured
            if self.config.include_metadata:
                for key, value in metadata.items():
                    if key not in chunk_metadata:
                        chunk_metadata[key] = value
            
            # Create ChunkMetadata object
            chunk_meta = ChunkMetadata(**chunk_metadata)
            
            # Create Chunk object
            chunk = Chunk(text=chunk_text, metadata=chunk_meta)
            result.append(chunk)
            
        return result
    
    def chunk_page(self, page: Page) -> List[Chunk]:
        """
        Split a Page object into chunks
        
        Args:
            page: The Page object to chunk
            
        Returns:
            List of Chunk objects
        """
        if not page.content or page.content.strip() == "":
            return []
            
        # Convert PageMeta to dictionary for the chunker
        page_metadata = page.meta.model_dump()
        
        # Add page ID if it exists
        if hasattr(page, "id"):
            page_metadata["id"] = page.id
            
        return self.chunk_text(page.content, page_metadata)
    
    def chunk_pages(self, pages: List[Page]) -> List[Chunk]:
        """
        Split multiple Page objects into chunks
        
        Args:
            pages: List of Page objects to chunk
            
        Returns:
            List of Chunk objects from all pages
        """
        all_chunks = []
        
        for page in pages:
            chunks = self.chunk_page(page)
            all_chunks.extend(chunks)
            
        return all_chunks
    
    def _chunk_by_size(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks of approximately equal size with overlap
        
        Returns:
            List of tuples (chunk_text, start_pos, end_pos)
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate the end position for this chunk
            end = start + self.config.chunk_size
            
            # Adjust end position if it exceeds text length
            if end > text_length:
                end = text_length
            elif end < text_length:
                # Try to find a good break point near the target end position
                # Look for a separator after the ideal end position
                next_separator = text.find(self.config.separator, end)
                
                # If found and not too far away, use it
                if next_separator != -1 and next_separator - end < 100:
                    end = next_separator + len(self.config.separator)
                else:
                    # Otherwise, look for the last separator before the end position
                    last_separator = text.rfind(self.config.separator, start, end)
                    if last_separator != -1 and end - last_separator < 100:
                        end = last_separator + len(self.config.separator)
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append((chunk, start, end))
            
            # Move the starting position for the next chunk,
            # accounting for overlap
            start = end - self.config.chunk_overlap
            
            # Ensure we make progress
            if start <= 0 or start >= text_length:
                break
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks respecting paragraph boundaries when possible
        
        Returns:
            List of tuples (chunk_text, start_pos, end_pos)
        """
        # Split the text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        start_pos = 0
        current_start = 0
        
        for para in paragraphs:
            para_size = len(para) + 2  # +2 for the newline characters
            
            # If adding this paragraph exceeds the chunk size and we already have content,
            # finalize the current chunk
            if current_size + para_size > self.config.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append((chunk_text, current_start, start_pos))
                
                # Start a new chunk with overlap
                # Find paragraphs that should be included due to overlap
                overlap_size = 0
                overlap_paragraphs = []
                
                for p in reversed(current_chunk):
                    p_size = len(p) + 2
                    if overlap_size + p_size <= self.config.chunk_overlap:
                        overlap_paragraphs.insert(0, p)
                        overlap_size += p_size
                    else:
                        break
                
                current_chunk = overlap_paragraphs
                current_size = overlap_size
                current_start = start_pos - overlap_size
            
            # Add the paragraph to the current chunk
            current_chunk.append(para)
            current_size += para_size
            start_pos += para_size
            
            # If this is the first paragraph in a chunk, update current_start
            if len(current_chunk) == 1:
                current_start = start_pos - para_size
        
        # Add the final chunk if there's anything left
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append((chunk_text, current_start, start_pos))
        
        return chunks
    
    def get_config(self) -> ChunkingConfig:
        """Get the current chunking configuration"""
        return self.config
    
    def set_config(self, config: ChunkingConfig) -> None:
        """Set a new chunking configuration"""
        self.config = config


# Usage example
if __name__ == "__main__":
    # Create a sample page
    from content_loader import Page, PageMeta
    
    meta = PageMeta(
        url="https://example.com/article",
        title="Sample Article",
        author="John Doe"
    )
    
    # Create a long sample text with multiple paragraphs
    sample_text = """
    This is the first paragraph of a sample document.
    It contains multiple sentences that will be part of the first chunk.
    
    This is the second paragraph that continues the document.
    It should still be included in the first chunk if the chunk size allows.
    
    This might be the start of a new chunk depending on the configuration.
    The chunker tries to respect paragraph boundaries when possible.
    
    """ + "\n\n".join([f"This is paragraph {i+4} with some content." for i in range(10)])
    
    page = Page(meta=meta, content=sample_text)
    
    # Create chunker with custom configuration
    config = ChunkingConfig(
        chunk_size=500,
        chunk_overlap=100,
        respect_paragraph=True
    )
    
    chunker = TextChunker(config)
    
    # Chunk the page
    chunks = chunker.chunk_page(page)
    
    # Print chunks
    for chunk in chunks:
        print(f"Chunk ID: {chunk.id}")
        print(f"Chunk {chunk.metadata.chunk_index + 1} of {chunk.metadata.total_chunks}")
        print(f"Chars {chunk.metadata.start_char}-{chunk.metadata.end_char}")
        print(f"Size: {len(chunk.text)} chars")
        print("Text preview: " + chunk.text[:100] + "...")
        print("-" * 80)