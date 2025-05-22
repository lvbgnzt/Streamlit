import streamlit as st
from serpsearch import search_serpapi

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

query = st.text_input("Was möchtest du googeln?", "")

if query and SERPAPI_API_KEY:
    st.write(f"🔎 Ergebnisse bei Google für: *{query}*")
    data = search_serpapi(query, SERPAPI_API_KEY, selected_location)

    if "organic_results" in data:
        for result in data["organic_results"]:
            st.markdown(f"**[{result['title']}]({result['link']})**")
            st.write(result.get("snippet", "Kein Snippet verfügbar."))
        st.subheader("📦 JSON-Daten der Top 10 Ergebnisse")
        st.json(data["organic_results"][:10])
    else:
        st.warning("Keine Ergebnisse oder Fehler bei der API.")