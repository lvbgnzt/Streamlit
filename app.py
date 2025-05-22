import streamlit as st
from serpsearch import search_serpapi, fetch_markdown_from_url

# Standort-Auswahl Dropdown
location_options = {
    "Deutschland (de)": "Germany",
    "USA (us)": "United States",
    "Vereinigtes Königreich (uk)": "United Kingdom",
    "Frankreich (fr)": "France",
    "Spanien (es)": "Spain"
}
selected_location_label = st.selectbox("🌍 Standort wählen", list(location_options.keys()))
selected_location = location_options[selected_location_label]

# SerpAPI Key (ersetze durch deinen eigenen Schlüssel)
SERPAPI_API_KEY = st.text_input("🔑 SerpAPI API Key", type="password")

# Firecrawl API Key
FIRECRAWL_API_KEY = st.text_input("🔥 Firecrawl API Key", type="password")

query = st.text_input("Was möchtest du googeln?", "")

# Search button
start_search = st.button("🔍 Suche starten")

if start_search and query and SERPAPI_API_KEY and FIRECRAWL_API_KEY:
    st.write(f"🔎 Ergebnisse bei Google für: *{query}*")
    data = search_serpapi(query, SERPAPI_API_KEY, selected_location)

    if "organic_results" in data:
        st.subheader("📝 Markdown-Inhalte der Top 3 Links")
        for idx, result in enumerate(data["organic_results"][:3], start=1):
            firecrawl_result = fetch_markdown_from_url(result["link"], FIRECRAWL_API_KEY)
            st.markdown(f"## Position {idx}: [{result['title']}]({result['link']})")
            st.code(firecrawl_result.data.markdown, language="markdown")
    else:
        st.warning("Keine Ergebnisse oder Fehler bei der API.")