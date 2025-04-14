import httpx
from typing import List, Dict, Tuple, Optional, Union, Any, Literal, Callable
from pydantic import BaseModel, Field, computed_field
from bs4 import BeautifulSoup
import json
import csv
import os
import hashlib
from datetime import datetime


class PageMeta(BaseModel):
    url: str
    title: Optional[str] = None
    author: Optional[str] = None
    pub_time: Optional[str] = None
    mod_time: Optional[str] = None

    # Allow additional fields beyond the predefined ones
    model_config = {"extra": "allow"}

    @computed_field
    def id(self) -> str:
        """Generate a unique ID based on the URL"""
        return hashlib.md5(self.url.encode()).hexdigest()


class Page(BaseModel):
    meta: PageMeta
    content: Optional[str] = None

    @computed_field
    def id(self) -> str:
        """Forward the ID from meta for easier access"""
        return self.meta.id


class ErrorResponse(BaseModel):
    status_code: int
    url: str


def extract_content_from_urls(
    urls: List[str],
    content_selector: str = "main > div.max-width-container div.grid-col:nth-child(2)",
    meta_extractors: Optional[Dict[str, Union[str, Callable]]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Tuple[List[Page], List[ErrorResponse]]:
    """
    Extract content from a list of URLs with configurable extractors.

    Args:
        urls: List of URLs to fetch and parse
        content_selector: CSS selector for the main content
        meta_extractors: Dictionary mapping field names to either:
                        - CSS selector strings (e.g., "title" or "meta[property='og:title']")
                        - Callable functions that take (soup, url) and return a value
        headers: Optional HTTP headers for the requests
        timeout: Optional timeout for HTTP requests in seconds

    Returns:
        Tuple of (list of Page objects, list of ErrorResponse objects)
    """
    non_200: List[ErrorResponse] = []
    pages: List[Page] = []

    # Default meta extractors if none provided
    if meta_extractors is None:
        meta_extractors = {
            "title": "title",
            "author": "meta[property='article:author']",
            "pub_time": "meta[property='article:published_time']",
            "mod_time": "meta[property='article:modified_time']",
        }

    # Default empty dict for headers if none provided
    if headers is None:
        headers = {}

    for url in urls:
        try:
            r = httpx.get(url, headers=headers, timeout=timeout)
            soup = BeautifulSoup(r.text, "html.parser")

            if r.status_code == 200:
                # Extract metadata
                meta_data = {"url": url}

                # Process all meta extractors
                for field, extractor in meta_extractors.items():
                    if isinstance(extractor, str):
                        # It's a CSS selector
                        if extractor.startswith("meta["):
                            # Handle meta tags
                            element = soup.select_one(extractor)
                            meta_data[field] = element["content"] if element else None
                        else:
                            # Handle regular elements
                            element = soup.select_one(extractor)
                            meta_data[field] = element.text.strip() if element else None
                    else:
                        # It's a callable extractor function
                        meta_data[field] = extractor(soup, url)

                # Create PageMeta with extracted data
                meta = PageMeta(**meta_data)

                # Extract content using provided selector
                content_element = soup.css.select_one(content_selector)
                content = content_element.text if content_element else None

                page = Page(meta=meta, content=content)
                pages.append(page)
            else:
                error = ErrorResponse(status_code=r.status_code, url=url)
                non_200.append(error)

        except Exception as e:
            # Handle any exceptions during fetching or parsing
            error = ErrorResponse(status_code=0, url=f"{url} (Error: {str(e)})")
            non_200.append(error)

    return pages, non_200


def save_content_to_disk(
    pages: List[Page],
    save_format: Literal["json", "csv"] = "json",
    output_dir: str = "./data/",
    include_computed: bool = True,
) -> str:
    """
    Save extracted content to disk in either JSON or CSV format.

    Args:
        pages: List of Page objects to save
        save_format: "json" or "csv" output format
        output_dir: Directory to save the output file
        include_computed: Whether to include computed fields (@computed_field) in the output

    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/pages_{timestamp}.{save_format}"

    if save_format.lower() == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            # Convert Pydantic models to JSON
            json.dump(
                [page.model_dump(by_alias=True, exclude_none=False) for page in pages],
                f,
                ensure_ascii=False,
                indent=2,
            )

    if save_format.lower() == "csv":
        # Collect all possible field names from all pages
        field_names = set()
        flattened_pages = []

        for page in pages:
            # Get the base model data
            page_dict = page.model_dump(by_alias=True, exclude_none=False)

            # Add computed fields if requested
            if include_computed and hasattr(page, "id"):
                page_dict["id"] = page.id

            # Flatten the nested structure (meta -> top level)
            flattened_page = {}

            # Handle meta fields
            for meta_key, meta_value in page_dict.get("meta", {}).items():
                flat_key = f"meta_{meta_key}"

                # Handle non-scalar values
                if isinstance(meta_value, (dict, list)):
                    flattened_page[flat_key] = json.dumps(meta_value)
                else:
                    flattened_page[flat_key] = meta_value

                # Add computed fields from meta if requested
                if include_computed and hasattr(page.meta, "id"):
                    flattened_page["meta_id"] = page.meta.id

            # Handle non-meta fields
            for key, value in page_dict.items():
                if key != "meta":
                    if isinstance(value, (dict, list)):
                        flattened_page[key] = json.dumps(value)
                    else:
                        flattened_page[key] = value

            # Collect all field names
            field_names.update(flattened_page.keys())
            flattened_pages.append(flattened_page)

        # Convert field_names set to sorted list for deterministic column order
        field_names = sorted(field_names)

        # Write CSV with all discovered fields
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(flattened_pages)

    return output_path


def load_content(file_path: str) -> List[Page]:
    """
    Load content from a previously saved file.

    Args:
        file_path: Path to the JSON or CSV file to load

    Returns:
        List of Page objects
    """
    # Determine file type from extension
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".json":
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return [Page.model_validate(item) for item in data]

    elif file_extension == ".csv":
        pages = []
        with open(file_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                # Extract fields with 'meta_' prefix for PageMeta
                meta_data = {}
                content = None
                other_data = {}

                for key, value in row.items():
                    # Try to parse lists and dicts from stringified JSON
                    try:
                        if value and (value.startswith("[") or value.startswith("{")):
                            value = json.loads(value)
                    except (json.JSONDecodeError, AttributeError):
                        pass

                    if key.startswith("meta_"):
                        # Remove 'meta_' prefix and add to meta_data
                        meta_key = key[5:]  # Remove 'meta_' prefix
                        meta_data[meta_key] = value
                    elif key == "content":
                        content = value
                    else:
                        other_data[key] = value

                # Create the Page object
                meta = PageMeta(**meta_data)
                page = Page(meta=meta, content=content, **other_data)
                pages.append(page)

        return pages

    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. Use .json or .csv files."
        )
