

from serpapi import GoogleSearch

def search_serpapi(query, api_key, location):
    params = {
        "q": query,
        "location": location,
        "hl": "de",
        "gl": "de",
        "google_domain": "google.de",
        "api_key": api_key
    }
    search = GoogleSearch(params)
    return search.get_dict()