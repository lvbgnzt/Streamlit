import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
import time
©
# Import our modules
from sitemap_loader import get_urls_from_sitemap
from content_loader import (
    extract_content_from_urls,
    save_content_to_disk,
    load_content,
    Page,
    PageMeta,
)
from text_chunker import TextChunker, ChunkingConfig, Chunk
from vector_store import VectorStore, ChromaConfig, QueryResult
from ai_chat import ChatEngine, RetrievalConfig

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Vector Search System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define app state keys
STATE_KEYS = {
    "urls": "extracted_urls",
    "pages": "extracted_pages",
    "chunks": "text_chunks",
    "vector_store": "vector_store_info",
    "chat_history": "chat_history",
}

# Initialize session state
for key in STATE_KEYS.values():
    if key not in st.session_state:
        st.session_state[key] = None

if "current_tab" not in st.session_state:
    st.session_state["current_tab"] = "Home"

if "pages_df" not in st.session_state:
    st.session_state["pages_df"] = None

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []


# Helper functions
def show_success_message(message: str):
    st.success(message, icon="✅")
    time.sleep(1)  # Keep message visible briefly


def show_info(title: str, data: Any):
    with st.expander(title, expanded=False):
        st.json(data)


def get_count(items) -> int:
    if items is None:
        return 0
    return len(items)


def format_chat_message(role, message):
    if role == "user":
        return st.chat_message(role, avatar="🧑‍💻").write(message)
    elif role == "assistant":
        return st.chat_message(role, avatar="🤖").write(message)
    else:
        return st.chat_message(role).write(message)


def clear_chat_history():
    st.session_state["chat_messages"] = []
    st.session_state["chat_engine"].reset_conversation()


def handle_chat_input():
    """Handle user input in the chat interface"""
    if st.session_state["chat_input"]:
        user_message = st.session_state["chat_input"]
        
        # Store the user message in session state
        if "last_user_message" not in st.session_state:
            st.session_state["last_user_message"] = user_message
            
        # We'll manage updating the chat history in the main app flow
        # to avoid rerun conflicts
        st.session_state["message_pending"] = True


# Navigation sidebar
st.sidebar.title("Navigation")
tabs = [
    "Home",
    "URL Extraction",
    "Content Loading",
    "Text Chunking",
    "Vector Storage",
    "Chat",
]
selected_tab = st.sidebar.radio("Go to", tabs, key="current_tab")

# Status indicators
st.sidebar.divider()
st.sidebar.subheader("Progress")
status_cols = st.sidebar.columns(2)
with status_cols[0]:
    st.write("URLs:")
    st.write("Pages:")
    st.write("Chunks:")
    st.write("Vector Store:")
with status_cols[1]:
    num_urls = get_count(st.session_state[STATE_KEYS["urls"]])
    st.write(f"**{num_urls}**")

    num_pages = get_count(st.session_state[STATE_KEYS["pages"]])
    st.write(f"**{num_pages}**")

    num_chunks = get_count(st.session_state[STATE_KEYS["chunks"]])
    st.write(f"**{num_chunks}**")

    if st.session_state[STATE_KEYS["vector_store"]]:
        vs_count = st.session_state[STATE_KEYS["vector_store"]].get("count", 0)
        st.write(f"**{vs_count}**")
    else:
        st.write("**0**")

# API Key management
st.sidebar.divider()
st.sidebar.subheader("API Keys")
openai_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.environ.get("OPENAI_API_KEY", ""),
    type="password",
    help="Your OpenAI API key for chat interactions",
)
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

# Environment settings
st.sidebar.divider()
st.sidebar.subheader("Settings")
data_dir = st.sidebar.text_input(
    "Data Directory", value="./data", help="Directory to save extracted content"
)
vector_db_dir = st.sidebar.text_input(
    "Vector DB Directory", value="./chroma_db", help="Directory for the vector database"
)

# Home Tab
if selected_tab == "Home":
    st.title("🔍 Vector Search System")

    st.markdown(
        """
    ## Welcome to the Vector Search System
    
    This application provides an end-to-end workflow for:
    1. 🌐 **Extracting URLs** from sitemaps
    2. 📄 **Loading content** from web pages
    3. ✂️ **Chunking text** into manageable pieces
    4. 🧠 **Creating vector embeddings** for similarity search
    5. 💬 **Chatting with your data** using AI
    
    ### Getting Started
    
    Follow these steps:
    1. Navigate to **URL Extraction** and input a sitemap URL
    2. Go to **Content Loading** to extract and process the content
    3. Use **Text Chunking** to split content into optimal sizes
    4. Create a **Vector Storage** database for similarity search
    5. Finally, **Chat** with your data using AI
    
    ### API Key Requirements
    
    This app requires:
    - An **OpenAI API key** for the chat functionality
    
    Enter your API keys in the sidebar to get started.
    """
    )

    # Dashboard cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**URLs Extracted**", icon="🌐")
        st.subheader(f"{num_urls}")
        if num_urls > 0:
            st.button(
                "View URLs",
                on_click=lambda: st.session_state.update(
                    {"current_tab": "URL Extraction"}
                ),
            )

    with col2:
        st.info("**Pages Loaded**", icon="📄")
        st.subheader(f"{num_pages}")
        if num_pages > 0:
            st.button(
                "View Pages",
                on_click=lambda: st.session_state.update(
                    {"current_tab": "Content Loading"}
                ),
            )

    with col3:
        st.info("**Vector Store**", icon="🧠")
        if st.session_state[STATE_KEYS["vector_store"]]:
            vs_count = st.session_state[STATE_KEYS["vector_store"]].get("count", 0)
            st.subheader(f"{vs_count} entries")
            if vs_count > 0:
                st.button(
                    "Start Chatting",
                    on_click=lambda: st.session_state.update({"current_tab": "Chat"}),
                )
        else:
            st.subheader("0 entries")

# URL Extraction Tab
elif selected_tab == "URL Extraction":
    st.title("🌐 URL Extraction")

    st.markdown(
        """
    Extract URLs from a sitemap.xml file. You can optionally filter URLs to include only those 
    matching a specific pattern.
    """
    )

    with st.form("sitemap_form"):
        sitemap_url = st.text_input(
            "Sitemap URL", placeholder="https://example.com/sitemap.xml"
        )
        filter_text = st.text_input(
            "Filter URLs containing (optional)", placeholder="blog"
        )
        submit_sitemap = st.form_submit_button("Extract URLs")

    if submit_sitemap and sitemap_url:
        with st.spinner("Extracting URLs from sitemap..."):
            try:
                urls = get_urls_from_sitemap(sitemap_url, filter_text or "")
                st.session_state[STATE_KEYS["urls"]] = urls
                st.success(f"Successfully extracted {len(urls)} URLs from the sitemap")
            except Exception as e:
                st.error(f"Error extracting URLs: {str(e)}")

    # Display extracted URLs
    if st.session_state[STATE_KEYS["urls"]]:
        st.subheader(f"{len(st.session_state[STATE_KEYS['urls']])} URLs Extracted")

        # Create a DataFrame for better display
        urls_df = pd.DataFrame({"URL": st.session_state[STATE_KEYS["urls"]]})
        st.dataframe(urls_df, use_container_width=True)

        # Export button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Export URLs to CSV"):
                urls_df.to_csv(f"{data_dir}/extracted_urls.csv", index=False)
                st.download_button(
                    label="Download URLs CSV",
                    data=urls_df.to_csv(index=False).encode("utf-8"),
                    file_name="extracted_urls.csv",
                    mime="text/csv",
                )

# Content Loading Tab
elif selected_tab == "Content Loading":
    st.title("📄 Content Loading")

    st.markdown(
        """
    Load content from the extracted URLs. The system will fetch each page, 
    extract the relevant content, and save it for further processing.
    """
    )

    # Check if we have URLs
    if not st.session_state[STATE_KEYS["urls"]]:
        st.warning(
            "No URLs have been extracted yet. Please go to the URL Extraction tab first."
        )
        if st.button("Go to URL Extraction"):
            st.session_state["current_tab"] = "URL Extraction"
            st.experimental_rerun()
    else:
        # Options for content extraction
        st.subheader("Content Extraction Options")

        with st.form("content_extraction_form"):
            num_urls = len(st.session_state[STATE_KEYS["urls"]])

            col1, col2 = st.columns(2)

            with col1:
                # URL limit slider
                max_urls = st.slider(
                    "Maximum URLs to process",
                    min_value=1,
                    max_value=min(100, num_urls),
                    value=min(10, num_urls),
                    help="Limit the number of URLs to process to avoid long processing times",
                )

            with col2:
                use_max_urls = st.checkbox(
                    "Process all URLs",
                    value=False,
                    help="Use with caution: This will process all urls and may take some time!",
                )

            if use_max_urls:
                max_urls = num_urls

            content_selector = st.text_input(
                "Content CSS Selector",
                value="main > div.max-width-container div.grid-col:nth-child(2)",
                help="CSS selector to extract the main content",
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("Default Meta Selectors:")
                st.code(
                    """
title: title
author: meta[property='article:author']
pub_time: meta[property='article:published_time']
mod_time: meta[property='article:modified_time']
                """
                )

            with col2:
                custom_meta = st.text_area(
                    "Custom Meta Selectors (JSON)",
                    value="{}",
                    help="Add custom CSS selectors for metadata in JSON format: {'field_name': 'css_selector'}",
                )

            submit_extraction = st.form_submit_button("Extract Content")

        if submit_extraction:
            try:
                custom_meta_selectors = json.loads(custom_meta)

                # Create full meta_selectors dict
                meta_selectors = {
                    "title": "title",
                    "author": "meta[property='article:author']",
                    "pub_time": "meta[property='article:published_time']",
                    "mod_time": "meta[property='article:modified_time']",
                }
                meta_selectors.update(custom_meta_selectors)

                with st.spinner(f"Extracting content from {max_urls} URLs..."):
                    # Get subset of URLs
                    urls_to_process = st.session_state[STATE_KEYS["urls"]][:max_urls]

                    # Extract content with progress bar
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    # Process in batches for better progress reporting
                    batch_size = 5
                    pages = []
                    errors = []

                    for i in range(0, len(urls_to_process), batch_size):
                        batch = urls_to_process[
                            i : min(i + batch_size, len(urls_to_process))
                        ]
                        progress_text.text(
                            f"Processing URLs {i+1}-{i+len(batch)} of {len(urls_to_process)}"
                        )

                        # Extract content with custom selectors
                        batch_pages, batch_errors = extract_content_from_urls(
                            urls=batch,
                            content_selector=content_selector,
                            meta_extractors=meta_selectors,
                        )

                        pages.extend(batch_pages)
                        errors.extend(batch_errors)

                        # Update progress
                        progress = min(1.0, (i + len(batch)) / len(urls_to_process))
                        progress_bar.progress(progress)

                    progress_bar.progress(1.0)
                    progress_text.text(f"Processed {len(urls_to_process)} URLs")

                    # Store in session state
                    st.session_state[STATE_KEYS["pages"]] = pages

                    # Create pages dataframe for display
                    pages_df = []
                    for page in pages:
                        # Handle both Pydantic model and dictionary formats
                        if hasattr(page, "meta") and hasattr(page, "content"):
                            # It's a Pydantic model (Page object)
                            page_dict = {
                                "url": page.meta.url,
                                "title": page.meta.title or "",
                                "content_length": (
                                    len(page.content) if page.content else 0
                                ),
                            }
                            # Add other meta fields
                            meta_dict = (
                                page.meta.model_dump()
                                if hasattr(page.meta, "model_dump")
                                else vars(page.meta)
                            )
                            for k, v in meta_dict.items():
                                if k not in ["url", "title"] and v is not None:
                                    page_dict[f"meta_{k}"] = v
                        else:
                            # It's a dictionary (old format)
                            page_dict = {
                                "url": page["meta"]["url"],
                                "title": page["meta"].get("title", ""),
                                "content_length": (
                                    len(page["content"]) if page["content"] else 0
                                ),
                            }
                            # Add other meta fields
                            for k, v in page["meta"].items():
                                if k not in ["url", "title"]:
                                    page_dict[f"meta_{k}"] = v

                        pages_df.append(page_dict)

                    if pages_df:
                        st.session_state["pages_df"] = pd.DataFrame(pages_df)

                    # Show results
                    st.success(
                        f"Successfully extracted content from {len(pages)} pages. {len(errors)} pages had errors."
                    )

                    # Save content
                    if pages:
                        output_file = save_content_to_disk(
                            pages, save_format="json", output_dir=data_dir
                        )
                        st.info(f"Content saved to: {output_file}")

                    # Display errors if any
                    if errors:
                        with st.expander("View Errors", expanded=False):
                            st.write(errors)

            except Exception as e:
                st.error(f"Error during content extraction: {str(e)}")

        # Display extracted content if available
        if st.session_state["pages_df"] is not None:
            st.subheader("Extracted Pages")
            st.dataframe(st.session_state["pages_df"], use_container_width=True)

            # Content preview
            if st.session_state[STATE_KEYS["pages"]]:
                with st.expander("Content Preview", expanded=False):
                    page_idx = st.selectbox(
                        "Select page to preview:",
                        range(len(st.session_state[STATE_KEYS["pages"]])),
                        format_func=lambda i: (
                            st.session_state["pages_df"]["title"].iloc[i]
                            if i < len(st.session_state["pages_df"])
                            else ""
                        ),
                    )

                    if page_idx is not None:
                        selected_page = st.session_state[STATE_KEYS["pages"]][page_idx]

                        # Handle both Pydantic model and dictionary formats
                        if hasattr(selected_page, "meta") and hasattr(
                            selected_page, "content"
                        ):
                            # It's a Pydantic model (Page object)
                            title = selected_page.meta.title or "Untitled"
                            url = selected_page.meta.url
                            content = (
                                selected_page.content[:1000] + "..."
                                if selected_page.content
                                else ""
                            )
                        else:
                            # It's a dictionary (old format)
                            title = selected_page["meta"].get("title", "Untitled")
                            url = selected_page["meta"]["url"]
                            content = (
                                selected_page["content"][:1000] + "..."
                                if selected_page["content"]
                                else ""
                            )

                        st.markdown(f"### {title}")
                        st.write(f"**URL:** {url}")
                        st.text_area(
                            "Content Preview:", value=content, height=300, disabled=True
                        )

        # Option to load from file
        st.divider()
        st.subheader("Or Load Content from File")

        with st.form("load_content_form"):
            content_file = st.text_input(
                "Path to content file:", f"{data_dir}/pages_*.json"
            )
            load_button = st.form_submit_button("Load Content")

        if load_button and content_file:
            try:
                import glob

                # Find the latest file matching the pattern
                matching_files = glob.glob(content_file)
                if not matching_files:
                    st.error(f"No files found matching: {content_file}")
                else:
                    # Sort by modification time (newest first)
                    latest_file = max(matching_files, key=os.path.getmtime)

                    with st.spinner(f"Loading content from {latest_file}..."):
                        pages = load_content(latest_file)
                        st.session_state[STATE_KEYS["pages"]] = pages

                        # Create dataframe for display
                        pages_df = []
                        for page in pages:
                            page_dict = {
                                "url": page.meta.url,
                                "title": page.meta.title or "",
                                "content_length": (
                                    len(page.content) if page.content else 0
                                ),
                            }
                            # Add other meta fields
                            for k, v in page.meta.model_dump().items():
                                if k not in ["url", "title"] and v is not None:
                                    page_dict[f"meta_{k}"] = v

                            pages_df.append(page_dict)

                        if pages_df:
                            st.session_state["pages_df"] = pd.DataFrame(pages_df)

                        st.success(
                            f"Successfully loaded {len(pages)} pages from {latest_file}"
                        )

            except Exception as e:
                st.error(f"Error loading content: {str(e)}")

# Text Chunking Tab
elif selected_tab == "Text Chunking":
    st.title("✂️ Text Chunking")

    st.markdown(
        """
    Split content into smaller chunks for better retrieval. You can configure chunk size,
    overlap, and other parameters to optimize for your specific content.
    """
    )

    # Check if we have pages
    if not st.session_state[STATE_KEYS["pages"]]:
        st.warning(
            "No content has been loaded yet. Please go to the Content Loading tab first."
        )
        if st.button("Go to Content Loading"):
            st.session_state["current_tab"] = "Content Loading"
            st.experimental_rerun()
    else:
        # Chunking configuration
        st.subheader("Chunking Configuration")

        with st.form("chunking_config_form"):
            col1, col2 = st.columns(2)

            with col1:
                chunk_size = st.slider(
                    "Chunk Size (characters)",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    help="Target size of each chunk in characters",
                )

                respect_paragraph = st.checkbox(
                    "Respect Paragraph Boundaries",
                    value=True,
                    help="Try to keep paragraphs together when possible",
                )

            with col2:
                chunk_overlap = st.slider(
                    "Chunk Overlap (characters)",
                    min_value=0,
                    max_value=500,
                    value=200,
                    help="Number of characters to overlap between chunks",
                )

                include_metadata = st.checkbox(
                    "Include Original Metadata",
                    value=True,
                    help="Include original page metadata with each chunk",
                )

            submit_chunking = st.form_submit_button("Create Text Chunks")

        if submit_chunking:
            with st.spinner("Creating text chunks..."):
                try:
                    # Configure chunker
                    config = ChunkingConfig(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        respect_paragraph=respect_paragraph,
                        include_metadata=include_metadata,
                    )

                    chunker = TextChunker(config)

                    # Get pages and ensure they're in the right format
                    pages = st.session_state[STATE_KEYS["pages"]]

                    # Convert dictionary-style pages to Page objects if needed
                    page_objects = []
                    for page in pages:
                        if hasattr(page, "meta"):
                            # Already a Page object
                            page_objects.append(page)
                        else:
                            # Convert dict to Page object
                            meta = PageMeta(**page["meta"])
                            page_obj = Page(meta=meta, content=page["content"])
                            page_objects.append(page_obj)

                    # Create chunks from page objects
                    chunks = chunker.chunk_pages(page_objects)

                    # Store in session state
                    st.session_state[STATE_KEYS["chunks"]] = chunks

                    st.success(
                        f"Successfully created {len(chunks)} chunks from {len(st.session_state[STATE_KEYS['pages']])} pages"
                    )

                except Exception as e:
                    st.error(f"Error creating chunks: {str(e)}")

        # Display chunks if available
        if st.session_state[STATE_KEYS["chunks"]]:
            chunks = st.session_state[STATE_KEYS["chunks"]]

            # Stats
            st.subheader("Chunk Statistics")

            # Calculate chunk length statistics
            chunk_lengths = [len(chunk.text) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)

            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Number of Chunks", len(chunks))
            with stat_col2:
                st.metric("Average Chunk Length", f"{avg_length:.1f} chars")
            with stat_col3:
                st.metric(
                    "Chunks per Page",
                    f"{len(chunks) / len(st.session_state[STATE_KEYS['pages']]):.1f}",
                )

            # Length distribution
            with st.expander("Chunk Length Distribution", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(chunk_lengths, bins=20)
                ax.set_xlabel("Chunk Length (characters)")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Chunk Lengths")
                st.pyplot(fig)

            # Chunk preview
            st.subheader("Chunk Preview")

            # Group chunks by source
            sources = {}
            for i, chunk in enumerate(chunks):
                source_id = chunk.metadata.source_id
                if source_id not in sources:
                    sources[source_id] = []
                sources[source_id].append(i)

            # Source selection
            selected_source = st.selectbox(
                "Select source document:",
                list(sources.keys()),
                format_func=lambda s: next(
                    (
                        (
                            p.meta.title
                            if hasattr(p, "meta")
                            else p["meta"].get("title", s)
                        )
                        for p in st.session_state[STATE_KEYS["pages"]]
                        if (hasattr(p, "id") and p.id == s)
                        or (not hasattr(p, "id") and p["meta"].get("id") == s)
                    ),
                    s,
                ),
            )

            if selected_source:
                # Chunk selection for the selected source
                chunk_indices = sources[selected_source]
                selected_chunk_idx = st.selectbox(
                    "Select chunk:",
                    chunk_indices,
                    format_func=lambda i: f"Chunk {chunks[i].metadata.chunk_index + 1} of {chunks[i].metadata.total_chunks} (Length: {len(chunks[i].text)} chars)",
                )

                # Display selected chunk
                if selected_chunk_idx is not None:
                    selected_chunk = chunks[selected_chunk_idx]

                    st.text_area(
                        "Chunk Content:",
                        value=selected_chunk.text,
                        height=300,
                        disabled=True,
                    )

                    # Chunk metadata
                    with st.expander("Chunk Metadata", expanded=False):
                        st.json(selected_chunk.metadata.model_dump())

# Vector Storage Tab
elif selected_tab == "Vector Storage":
    st.title("🧠 Vector Storage")

    st.markdown(
        """
    Create vector embeddings for your content and store them for similarity search.
    You can choose different embedding models and configure the vector store.
    """
    )

    # Check if we have pages or chunks
    has_content = (
        st.session_state[STATE_KEYS["pages"]] is not None
        or st.session_state[STATE_KEYS["chunks"]] is not None
    )

    if not has_content:
        st.warning(
            "No content has been loaded yet. Please go to the Content Loading or Text Chunking tab first."
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Content Loading"):
                st.session_state["current_tab"] = "Content Loading"
                st.experimental_rerun()
        with col2:
            if st.button("Go to Text Chunking"):
                st.session_state["current_tab"] = "Text Chunking"
                st.experimental_rerun()
    else:
        # Vector store configuration
        st.subheader("Vector Store Configuration")

        with st.form("vector_store_form"):
            collection_name = st.text_input("Collection Name", "document_collection")

            embedding_function = st.selectbox(
                "Embedding Function",
                ["default", "openai", "cohere"],
                help="Embedding model to use. default = local all-MiniLM-L6-v2 model",
            )

            use_chunks = st.radio(
                "Content to Store",
                ["Text Chunks", "Full Pages"],
                index=0 if st.session_state[STATE_KEYS["chunks"]] else 1,
                help="Store chunks (recommended) or full pages in the vector store",
            )

            submit_vector_store = st.form_submit_button("Create Vector Store")

        if submit_vector_store:
            with st.spinner("Creating vector store..."):
                try:
                    # Configure vector store
                    chroma_config = ChromaConfig(
                        persist_directory=vector_db_dir,
                        collection_name=collection_name,
                        embedding_function_name=embedding_function,
                    )

                    # Determine if we should use chunking
                    if (
                        use_chunks == "Text Chunks"
                        and st.session_state[STATE_KEYS["chunks"]]
                    ):
                        # Using pre-created chunks
                        chunking_config = None  # No need to create chunks again
                        vector_store = VectorStore(chroma_config, chunking_config)

                        # Add chunks directly
                        chunks = st.session_state[STATE_KEYS["chunks"]]
                        vector_store.add_chunks(chunks)

                    else:
                        # Using full pages or generate chunks on-the-fly
                        if use_chunks == "Text Chunks":
                            # Create chunks on-the-fly
                            chunking_config = ChunkingConfig(
                                chunk_size=1000,
                                chunk_overlap=200,
                                respect_paragraph=True,
                            )
                            vector_store = VectorStore(chroma_config, chunking_config)
                        else:
                            # Use full pages without chunking
                            chunking_config = None
                            vector_store = VectorStore(chroma_config, chunking_config)

                        # Get pages and ensure they're in the right format
                        pages = st.session_state[STATE_KEYS["pages"]]

                        # Convert dictionary-style pages to Page objects if needed
                        page_objects = []
                        for page in pages:
                            if hasattr(page, "meta"):
                                # Already a Page object
                                page_objects.append(page)
                            else:
                                # Convert dict to Page object
                                meta = PageMeta(**page["meta"])
                                page_obj = Page(meta=meta, content=page["content"])
                                page_objects.append(page_obj)

                        # Add pages
                        vector_store.add_pages(page_objects)

                    # Get collection info
                    collection_info = vector_store.get_collection_info()

                    # Store in session state
                    st.session_state[STATE_KEYS["vector_store"]] = collection_info

                    st.success(
                        f"Successfully created vector store with {collection_info['count']} entries"
                    )

                except Exception as e:
                    st.error(f"Error creating vector store: {str(e)}")

        # Vector store info if available
        if st.session_state[STATE_KEYS["vector_store"]]:
            collection_info = st.session_state[STATE_KEYS["vector_store"]]

            # Stats
            st.subheader("Vector Store Information")

            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Total Entries", collection_info["count"])
            with info_col2:
                st.metric("Collection Name", collection_info["name"])
            with info_col3:
                st.metric("Embedding Model", collection_info["embedding_function"])

            st.info(f"Database Location: {collection_info['persist_directory']}")

            # Search test
            st.subheader("Test Vector Search")

            test_query = st.text_input(
                "Enter a test query:", placeholder="What is vector search?"
            )

            if test_query:
                try:
                    # Create a temp vector store for testing
                    chroma_config = ChromaConfig(
                        persist_directory=collection_info["persist_directory"],
                        collection_name=collection_info["name"],
                        embedding_function_name=collection_info["embedding_function"],
                    )

                    test_store = VectorStore(chroma_config)

                    # Search
                    with st.spinner("Searching..."):
                        results = test_store.query(query_text=test_query, n_results=3)

                        if results:
                            st.success(f"Found {len(results)} results")

                            # Create tabs for results instead of nested expanders
                            result_tabs = st.tabs(
                                [
                                    f"Result {i+1} (Score: {r.score:.4f})"
                                    for i, r in enumerate(results)
                                ]
                            )

                            for i, (result, tab) in enumerate(
                                zip(results, result_tabs)
                            ):
                                with tab:
                                    # Title or source
                                    title = (
                                        result.metadata.get("title")
                                        or result.metadata.get("source_url")
                                        or "Untitled"
                                    )
                                    st.markdown(f"**{title}**")

                                    # Content
                                    st.text_area(
                                        f"Content:",
                                        value=result.document.get("content", ""),
                                        height=150,
                                        disabled=True,
                                    )

                                    # Show metadata in a collapsible section
                                    with st.expander("Metadata", expanded=False):
                                        st.json(result.metadata)
                        else:
                            st.warning("No results found for this query.")

                except Exception as e:
                    st.error(f"Error during search: {str(e)}")

# Chat Tab
elif selected_tab == "Chat":
    st.title("💬 Chat with Your Data")

    st.markdown(
        """
    Ask questions about your content and get AI-generated answers based on the
    information in your vector store.
    """
    )

    # Check if we have a vector store
    if not st.session_state[STATE_KEYS["vector_store"]]:
        st.warning(
            "No vector store has been created yet. Please go to the Vector Storage tab first."
        )
        if st.button("Go to Vector Storage"):
            st.session_state["current_tab"] = "Vector Storage"
            st.experimental_rerun()
    else:
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            st.error(
                "OpenAI API key is required for chat functionality. Please enter your API key in the sidebar."
            )
        else:
            # Chat interface
            st.subheader("Chat with Your Content")

            # Initialize chat if needed
            if "chat_engine" not in st.session_state:
                try:
                    # Create vector store with existing configuration
                    collection_info = st.session_state[STATE_KEYS["vector_store"]]
                    chroma_config = ChromaConfig(
                        persist_directory=collection_info["persist_directory"],
                        collection_name=collection_info["name"],
                        embedding_function_name=collection_info["embedding_function"],
                    )

                    vector_store = VectorStore(chroma_config)

                    # Create chat configuration
                    chat_config = RetrievalConfig(
                        model=st.sidebar.selectbox(
                            "Chat Model",
                            [
                                "gpt-3.5-turbo",
                                "gpt-3.5-turbo-16k",
                                "gpt-4o",
                                "gpt-4-turbo",
                            ],
                            index=0,
                            help="Select the model to use for chat",
                        ),
                        top_k=5,
                        similarity_threshold=0.7,
                        include_sources=True,
                        system_prompt="You are a helpful assistant that answers questions based on the provided content. Only use the information from the provided content and be honest when you don't know.",
                    )

                    # Create chat engine
                    st.session_state["chat_engine"] = ChatEngine(
                        vector_store, chat_config
                    )
                    st.success("Chat engine initialized successfully")

                except Exception as e:
                    st.error(f"Error initializing chat engine: {str(e)}")

            # Chat settings
            with st.expander("Chat Settings", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    # Update model if changed
                    new_model = st.selectbox(
                        "Chat Model",
                        ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4o", "gpt-4-turbo"],
                        index=(
                            [
                                "gpt-3.5-turbo",
                                "gpt-3.5-turbo-16k",
                                "gpt-4o",
                                "gpt-4-turbo",
                            ].index(st.session_state["chat_engine"].config.model)
                            if "chat_engine" in st.session_state
                            else 0
                        ),
                    )

                    if (
                        "chat_engine" in st.session_state
                        and new_model != st.session_state["chat_engine"].config.model
                    ):
                        st.session_state["chat_engine"].config.model = new_model

                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        help="Higher values make output more random",
                    )

                    if "chat_engine" in st.session_state:
                        st.session_state["chat_engine"].config.temperature = temperature

                with col2:
                    top_k = st.slider(
                        "Number of chunks to retrieve",
                        min_value=1,
                        max_value=10,
                        value=5,
                        help="Number of most similar chunks to use for context",
                    )

                    similarity_threshold = st.slider(
                        "Similarity Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.05,
                        help="Minimum similarity score to include a chunk",
                    )

                    if "chat_engine" in st.session_state:
                        st.session_state["chat_engine"].config.top_k = top_k
                        st.session_state["chat_engine"].config.similarity_threshold = (
                            similarity_threshold
                        )

                include_sources = st.checkbox(
                    "Include source information in responses", value=True
                )

                if "chat_engine" in st.session_state:
                    st.session_state["chat_engine"].config.include_sources = (
                        include_sources
                    )

                if st.button("Clear Chat History"):
                    st.session_state["chat_messages"] = []
                    if "chat_engine" in st.session_state:
                        st.session_state["chat_engine"].reset_conversation()
                    st.experimental_rerun()

            # First check if we have a pending message to process
            if st.session_state.get("message_pending", False) and "last_user_message" in st.session_state:
                # Get the message that needs to be processed
                user_message = st.session_state["last_user_message"]
                
                # Add the user message to chat history
                st.session_state["chat_messages"].append(
                    {"role": "user", "content": user_message}
                )
                
                # Set up a spinner while generating the response
                with st.spinner("Generating response..."):
                    # Get the response non-streaming to avoid UI flicker
                    full_response = st.session_state["chat_engine"].ask(user_message)
                    
                    # Add the complete response to chat history
                    st.session_state["chat_messages"].append(
                        {"role": "assistant", "content": full_response}
                    )
                
                # Reset the pending message flags
                st.session_state["message_pending"] = False
                del st.session_state["last_user_message"]
                
                # The page will automatically rerun after this due to the state change
            
            # Display chat messages from history with proper formatting
            messages = st.session_state["chat_messages"]
            
            # Message container to keep all messages together
            with st.container():
                for i, message in enumerate(messages):
                    # Get the message content and role
                    role = message["role"]
                    content = message["content"]
                    
                    # Display with appropriate avatar and styling
                    if role == "user":
                        with st.chat_message(role, avatar="🧑‍💻"):
                            st.markdown(content)
                    elif role == "assistant":
                        with st.chat_message(role, avatar="🤖"):
                            st.markdown(content)
                    else:
                        with st.chat_message(role):
                            st.markdown(content)

            # Chat input
            if "chat_engine" in st.session_state:
                st.chat_input(
                    "Ask a question about your content...",
                    key="chat_input",
                    on_submit=handle_chat_input,
                )

# Run the app
if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vector_db_dir, exist_ok=True)
