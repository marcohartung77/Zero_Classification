import streamlit as st
from transformers import pipeline
import pandas as pd
import requests
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

@st.cache_data
def google_news_rss(query, lang="de", country="DE", max_items=20, timeout=10):
    url = (
        f"https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl={lang}&gl={country}&ceid={country}:{lang}"
    )
    
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    
    items = []
    root = ET.fromstring(response.content)
    for item in root.findall(".//item")[:max_items]:
        item_title = item.findtext("title")
        item_pub_date = item.findtext("pubDate")
        item_desc = item.findtext("description")
        items.append({
            "title": item_title,
            "date": item_pub_date,
            "description": item_desc
        })
    return items

@st.cache_resource
def load_classifier(model_name: str):
    return pipeline(task="zero-shot-classification", model=model_name)

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

# Daten für Session State:
if "news_topic" not in st.session_state:
    st.session_state.news_topic = "Bitcoin"

news_list = google_news_rss(query=st.session_state.news_topic)
news_titles = []
news_dates = []
date_title_list = []
for news in news_list:
    news_titles.append(news["title"])
    news_dates.append(news["date"])
    formated_date = news['date'].split(" ")[1:4]
    short_date = " ".join(formated_date)
    date_title_list.append(f"{short_date} - {news['title']}")

if date_title_list:
    st.session_state.date_title_list = date_title_list

if "textarea_value" not in st.session_state:
    st.session_state.textarea_value = ""

st.title("Zero-Shot-Klassifikation")
st.caption("Klassifiziere beliebige Texte in selbst gewählte Kategorien")

if "categories_input" not in st.session_state:
    st.session_state.categories_input = "economy, politics, sport, entertainment, health, finance"

model_choice = st.selectbox(label="Model auswählen", options=["facebook/bart-large-mnli"])
clf = load_classifier(model_choice)

st.subheader("Texteingabe")
st.session_state.news_topic = st.text_input(
    label="Gib eine Thematik ein, für die Nachrichtenabfrage:",
    value=st.session_state.news_topic
    )

if st.button("Nachrichten laden"):
    news_list = google_news_rss(query=st.session_state.news_topic)
    news_titles = []
    news_dates = []
    date_title_list = []
    for news in news_list:
        news_titles.append(news["title"])
        news_dates.append(news["date"])
        formated_date = news['date'].split(" ")[1:4]
        short_date = " ".join(formated_date)
        date_title_list.append(f"{short_date} - {news['title']}")

selectbox_value = st.selectbox(label="Nachrichtentitel auswählen", options=st.session_state.date_title_list)

if st.button("Google Nachrichten verwenden"):
    title_without_date = selectbox_value.split(" - ", 1)[-1]
    st.session_state.textarea_value = title_without_date
    
text = st.text_area(
    label="Gib hier deinen Text ein:",
    placeholder="Beispiel: Die Inflation ist im September erneut gestiegen....",
    value=st.session_state.textarea_value
)

st.subheader("Kategorieneingabe")
st.write("Vorauswahl, falls nötig treffen:")

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("Nachrichten"):
        categories_input = "economy, politics, sport, entertainment, health, finance"
        st.session_state.categories_input = categories_input       

with col2:
    if st.button("Bildung"):
        categories_input = "education, school, university, learning, e-learning, curriculum"
        st.session_state.categories_input = categories_input

categories_input = st.text_input(
    label="Kategorien durch Komma getrennt eingeben:",
    key="categories_input",
    help="Kategorien können beliebig gewählt werden durch Texteingabe"
)

multi_label = st.checkbox(label="Mehrere Kategorien zulassen")

if st.button("Klassifizieren"):
    if not text:
        st.warning("Bitte gib einen Text ein")
    else:
        raw_parts = categories_input.split(",")
        labels = []
        for raw_part in raw_parts:
            cleaned = raw_part.strip().replace('"', '')
            labels.append(cleaned)
        if len(labels) < 2:
            st.warning("Bitte mindestens zwei Kategorien angeben")
        else:
            with st.spinner("Modell berechnet..."):
                res = clf(text, candidate_labels=labels, multi_label=multi_label)
            
            df = pd.DataFrame({
                "Kategorie": res["labels"],
                "Wahrscheinlichkeit": pd.Series(res["scores"]) * 100
            })
            
            st.subheader("Resultate")
            top_cat = df.iloc[0]["Kategorie"]
            st.write(f"Die wahrscheinlchste Kategorie ist: {top_cat}")
            st.dataframe(data=df.style.format({"Wahrscheinlichkeit": "{:.2f}%"}))
            csv = convert_df(df)
            st.download_button(
                "Kategorisierte Daten Herunterladen",
                csv,
                "file.csv",
                "text/csv",
                key='download-csv'
            )
            st.bar_chart(data=df.set_index("Kategorie")["Wahrscheinlichkeit"])
