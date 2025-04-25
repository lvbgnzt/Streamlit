import os
from typing import List, Dict, Any, Optional, Union, Tuple, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from vector_store import VectorStore, QueryResult
from content_loader import Page
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Message(BaseModel):
    """Chat message model"""

    role: Literal["system", "user", "assistant", "function", "tool"] = "user"
    content: str
    name: Optional[str] = None


class ChatHistory(BaseModel):
    """Track conversation history"""

    messages: List[Message] = []

    def add_message(self, role: str, content: str, name: Optional[str] = None) -> None:
        """Add a message to the history"""
        self.messages.append(Message(role=role, content=content, name=name))

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get messages in format expected by OpenAI API"""
        return [message.model_dump(exclude_none=True) for message in self.messages]

    def clear(self) -> None:
        """Clear the conversation history"""
        self.messages = []


class RetrievalConfig(BaseModel):
    """Configuration for retrieval-augmented conversations"""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_k: int = 5  # Number of documents to retrieve
    similarity_threshold: float = 0.7  # Minimum similarity score to include
    include_sources: bool = True  # Whether to include source information
    system_prompt: str = (
        "You are a helpful assistant that answers questions based on the provided context."
    )


class ChatEngine:
    """Manages conversations with OpenAI models using vector store for retrieval"""

    def __init__(
        self,
        vector_store: VectorStore,
        config: RetrievalConfig = RetrievalConfig(),
        api_key: Optional[str] = None,
    ):
        """
        Initialize the chat engine

        Args:
            vector_store: Vector store for document retrieval
            config: Configuration for the chat engine
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        """
        self.vector_store = vector_store
        self.config = config
        self.history = ChatHistory()

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        # Set up system message
        self.history.add_message("system", self.config.system_prompt)

    def ask(self, query: str) -> str:
        """
        Ask a question and get a response based on retrieved documents

        Args:
            query: The user's question

        Returns:
            The AI's response text
        """
        # Add user query to history
        self.history.add_message("user", query)

        # Retrieve relevant documents
        relevant_docs = self._retrieve_documents(query)

        # Format context from documents
        context = self._format_context(relevant_docs)

        # Add context as a system message
        if context:
            context_message = f"Use the following information to answer the user's question:\n\n{context}"
            self.history.add_message("system", context_message)

        # Get response from OpenAI
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=self.history.get_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
        )

        # Extract content
        response_text = response.choices[0].message.content

        # Add response to history
        self.history.add_message("assistant", response_text)

        return response_text

    def ask_stream(self, query: str) -> Any:
        """
        Ask a question and get a streaming response

        Args:
            query: The user's question

        Returns:
            A generator that yields response chunks
        """
        # Add user query to history
        self.history.add_message("user", query)

        # Retrieve relevant documents
        relevant_docs = self._retrieve_documents(query)

        # Format context from documents
        context = self._format_context(relevant_docs)

        # Add context as a system message
        if context:
            context_message = f"Use the following information to answer the user's question:\n\n{context}"
            self.history.add_message("system", context_message)

        # Get streaming response from OpenAI
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=self.history.get_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        # Collect the full response as we stream it
        full_response = ""

        # Stream the response
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # Add completed response to history
        self.history.add_message("assistant", full_response)

    def _retrieve_documents(self, query: str) -> List[QueryResult]:
        """
        Retrieve relevant documents based on the query

        Args:
            query: The search query

        Returns:
            List of relevant documents
        """
        results = self.vector_store.query(query_text=query, n_results=self.config.top_k)

        # Filter by similarity threshold
        filtered_results = [
            result
            for result in results
            if result.score >= self.config.similarity_threshold
        ]

        return filtered_results

    def _format_context(self, documents: List[QueryResult]) -> str:
        """
        Format retrieved documents into a context string

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        context_parts = []

        for i, doc in enumerate(documents):
            # Extract content and metadata
            content = doc.document.get("content", "")

            # Format the document with optional source info
            if self.config.include_sources:
                # Get source information
                source_url = doc.metadata.get("source_url") or doc.metadata.get(
                    "url", "Unknown source"
                )
                title = doc.metadata.get("title", "Untitled document")

                # Check if this is a chunk
                if "chunk_index" in doc.metadata:
                    chunk_info = f" (Chunk {doc.metadata['chunk_index'] + 1} of {doc.metadata['total_chunks']})"
                else:
                    chunk_info = ""

                # Format source header
                source_info = f"Source {i+1}: {title} - {source_url}{chunk_info}"
                doc_text = f"{source_info}\n{content}\n"
            else:
                doc_text = f"Document {i+1}:\n{content}\n"

            context_parts.append(doc_text)

        return "\n".join(context_parts)

    def reset_conversation(self) -> None:
        """Reset the conversation history but keep the system prompt"""
        system_prompt = self.config.system_prompt
        self.history.clear()
        self.history.add_message("system", system_prompt)


# Example usage
if __name__ == "__main__":
    from vector_store import VectorStore, ChromaConfig
    from text_chunker import ChunkingConfig, TextChunker
    from content_loader import Page, PageMeta

    # Set up vector store with example data
    print("Setting up vector store...")
    chroma_config = ChromaConfig(
        persist_directory="./chroma_db",
        collection_name="chat_example",
        embedding_function_name="default",  # Use default for demo without API key
    )

    # Enable chunking
    chunking_config = ChunkingConfig(chunk_size=500, chunk_overlap=100)

    vector_store = VectorStore(chroma_config, chunking_config)

    # Check if we already have documents
    if vector_store.get_page_count() == 0:
        print("Adding sample documents...")
        # Create some sample documents
        pages = [
            Page(
                meta=PageMeta(
                    url="https://example.com/vector-search",
                    title="Introduction to Vector Search",
                ),
                content="""
                Vector search is a powerful technique for finding similar items in large datasets.
                It works by converting items into high-dimensional vectors and finding nearest neighbors.
                
                Unlike traditional keyword search, vector search can understand semantic similarity.
                This makes it ideal for applications like recommendation systems, image search, and natural language processing.
                
                Key benefits of vector search include:
                1. Finding similar items even when they don't share exact keywords
                2. Understanding conceptual relationships between items
                3. Supporting multimodal search across text, images, and other data types
                """,
            ),
            Page(
                meta=PageMeta(
                    url="https://example.com/embeddings",
                    title="Understanding Embeddings",
                ),
                content="""
                Embeddings are numerical representations of data in a continuous vector space.
                They capture semantic meaning and relationships between items.
                
                In NLP, word and sentence embeddings map language to vectors where similar meanings are close together.
                Modern embedding techniques like those in OpenAI's models can capture complex relationships between concepts.
                
                Creating good embeddings requires:
                1. Large amounts of training data
                2. Sophisticated neural network architectures
                3. Techniques to handle different languages and contexts
                """,
            ),
        ]

        # Add to vector store
        vector_store.add_pages(pages)

    # Create chat engine
    print("Initializing chat engine...")
    config = RetrievalConfig(
        model="gpt-3.5-turbo",  # Use 3.5 for cheaper testing
        top_k=3,
        system_prompt="You are a helpful AI assistant that answers questions based on the retrieved documents. Only use the information from the documents to answer questions.",
    )

    # Get API key from environment and print status
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "\nWARNING: OPENAI_API_KEY not found in environment variables or .env file."
        )
        print(
            "You need to create a .env file in the project root with your OpenAI API key:"
        )
        print("OPENAI_API_KEY=your-api-key-here")
        print(
            "\nAlternatively, you can provide the API key directly to the ChatEngine."
        )

        # Ask for API key input for convenience
        print("\nWould you like to enter your OpenAI API key manually? (y/n)")
        if input().lower() == "y":
            api_key = input("Enter your OpenAI API key: ").strip()
            print("Using provided API key for this session (will not be saved)")
        else:
            print("Exiting. Please set up your API key before running again.")
            exit()
    else:
        print("OPENAI_API_KEY found in environment.")

    try:
        # Create chat engine with optional manual API key
        chat_engine = ChatEngine(vector_store, config, api_key=api_key)

        # Example conversation
        print("\n--- Chat Example ---")
        print("You can ask questions about vector search and embeddings.")
        print("Type 'exit' to quit.")

        while True:
            user_input = input("\nYou: ")

            if user_input.lower() in ["exit", "quit"]:
                break

            print("\nAI: ", end="")

            # For streaming output
            for text_chunk in chat_engine.ask_stream(user_input):
                print(text_chunk, end="", flush=True)

            print()  # Add newline after response

    except Exception as e:
        error_message = str(e).lower()
        if "api_key" in error_message or "authentication" in error_message:
            print("\nError: OpenAI API key invalid or has insufficient permissions.")
            print(
                "Please check your API key and ensure it has the necessary permissions."
            )
        elif "billing" in error_message or "quota" in error_message:
            print(
                "\nError: OpenAI API billing issue. You may have exceeded your quota or need to set up billing."
            )
        else:
            print(f"\nError: {str(e)}")
