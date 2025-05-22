from serpapi import GoogleSearch

def search_serpapi(query, serp_api_key, location):
    params = {
        "q": query,
        "location": location,
        "hl": "de",
        "gl": "de",
        "google_domain": "google.de",
        "api_key": serp_api_key
    }
    search = GoogleSearch(params)
    return search.get_dict()

from firecrawl import FirecrawlApp

def fetch_markdown_from_url(url: str, firecrawl_api_key: str) -> str:
    app = FirecrawlApp(api_key=firecrawl_api_key)
    response = app.scrape_url(url, formats=['markdown'])
    if response.get("success") and "data" in response and "markdown" in response["data"]:
        return response["data"]["markdown"]
    return "Fehler beim Abrufen von Markdown."