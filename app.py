import streamlit as st
streamlit
requests
google-search-results

# SerpAPI Key (ersetze durch deinen eigenen Schlüssel)
SERPAPI_API_KEY = "DEIN_API_KEY"

def search_serpapi(query, api_key):
    params = {
        "q": query,
        "location": "Germany",
        "hl": "de",
        "gl": "de",
        "google_domain": "google.de",
        "api_key": api_key
    }
    search = GoogleSearch(params)
    return search.get_dict()

query = st.text_input("Was möchtest du googeln?", "")

if query:
    st.write(f"🔎 Ergebnisse bei Google für: *{query}*")
    data = search_serpapi(query, SERPAPI_API_KEY)

    if "organic_results" in data:
        for result in data["organic_results"]:
            st.markdown(f"**[{result['title']}]({result['link']})**")
            st.write(result.get("snippet", "Kein Snippet verfügbar."))
    else:
        st.warning("Keine Ergebnisse oder Fehler bei der API.")