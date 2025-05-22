import streamlit as st

st.set_page_config(page_title="🧠 Semantische Suche", layout="centered")

st.title("🔍 Semantische Suche (Demo)")
st.write("Dies ist eine simple Streamlit-App zur interaktiven Textsuche.")

# Beispielhafte "Dokumentdatenbank"
documents = [
    "Künstliche Intelligenz verändert die Welt.",
    "Python ist eine vielseitige Programmiersprache.",
    "Streamlit ermöglicht schnelle Prototypen für Daten-Apps.",
    "Vektordatenbanken speichern semantische Informationen.",
    "OpenAI entwickelt leistungsstarke Sprachmodelle."
]

# Nutzereingabe
query = st.text_input("Was möchtest du suchen?", "")

# "Suche" auslösen
if query:
    st.write(f"🔎 Ergebnisse für: *{query}*")

    # Platzhalter für spätere Vektorsuche: einfache Keyword-Matching-Demo
    matches = [doc for doc in documents if query.lower() in doc.lower()]

    if matches:
        for match in matches:
            st.success(match)
    else:
        st.warning("Keine Ergebnisse gefunden. Probiere einen anderen Begriff.")
else:
    st.info("Gib oben einen Suchbegriff ein.")
