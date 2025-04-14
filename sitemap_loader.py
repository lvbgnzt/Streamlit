import httpx
import xml.etree.ElementTree as ET
from typing import List


def get_urls_from_sitemap(sitemap_url: str, filter: str) -> List[str]:
    r = httpx.get(sitemap_url)
    root = ET.fromstring(r.text)

    # XML namespace in sitemap
    namespaces = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls: List[str] = []

    for url in root.findall(".//ns:url", namespaces):
        loc = url.find("ns:loc", namespaces)
        if loc is not None and loc.text is not None and filter in loc.text:
            urls.append(loc.text)

    return urls
