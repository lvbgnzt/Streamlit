import streamlit as st
from serpapi import GoogleSearch

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

def search_serpapi(query, api_key):
    params = {
        "q": query,
        "location": selected_location,
        "hl": "de",
        "gl": "de",
        "google_domain": "google.de",
        "api_key": api_key
    }
    search = GoogleSearch(params)
    return search.get_dict()

query = st.text_input("Was möchtest du googeln?", "")

if query and SERPAPI_API_KEY:
    st.write(f"🔎 Ergebnisse bei Google für: *{query}*")
    data = search_serpapi(query, SERPAPI_API_KEY)

    if "organic_results" in data:
        for result in data["organic_results"]:
            st.markdown(f"**[{result['title']}]({result['link']})**")
            st.write(result.get("snippet", "Kein Snippet verfügbar."))
    else:
        st.warning("Keine Ergebnisse oder Fehler bei der API.")