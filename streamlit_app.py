import streamlit as st
import pickle
import sqlite3
from textblob import TextBlob
from PIL import Image
import pandas as pd
import os
from wordcloud import WordCloud
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🕵️ Fake Review Detection System")

# ---------- DB Initialization ----------
def init_db():
    conn = sqlite3.connect('review_data.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT,
            prediction TEXT,
            confidence REAL,
            sentiment TEXT,
            date TEXT,
            admin_tag TEXT DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()
init_db()

# ---------- Load ML Model ----------
model = pickle.load(open("model.pkl", "rb"))

# ---------- Helper ----------
def highlight_keywords(text):
    spam_keywords = ["free", "click", "amazing", "deal", "guaranteed", "perfect", "refund", "buy now", "limited", "win"]
    for word in spam_keywords:
        text = text.replace(word, f"**:red[{word}]**")
    return text

def generate_wordcloud(texts):
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(texts))
    wc.to_file("wordcloud.png")

def tag_review(review_id, tag):
    conn = sqlite3.connect("review_data.db")
    c = conn.cursor()
    c.execute("UPDATE reviews SET admin_tag=? WHERE id=?", (tag, review_id))
    conn.commit()
    conn.close()
# ---------- Review Input ----------
review_text = st.text_area("✍️ Enter Review Text", height=150)

if st.button("🔍 Analyze Review"):
    if not review_text.strip():
        st.warning("Please enter a review.")
    else:
        prediction = model.predict([review_text])[0]
        confidence = model.predict_proba([review_text])[0][prediction]
        sentiment_score = TextBlob(review_text).sentiment.polarity

        sentiment_label = "Neutral"
        if sentiment_score > 0.2:
            sentiment_label = "Positive"
        elif sentiment_score < -0.2:
            sentiment_label = "Negative"

        word_count = len(review_text.split())

        st.subheader("🔎 Result")
        st.markdown(f"**Prediction:** {'🚨 FAKE' if prediction else '✅ GENUINE'}")
        st.markdown(f"**Confidence:** `{confidence:.2%}`")
        st.markdown(f"**Sentiment:** `{sentiment_label}`")
        st.markdown(f"**Word Count:** `{word_count}`")

        st.subheader("📌 Highlighted Review")
        st.markdown(highlight_keywords(review_text.lower()), unsafe_allow_html=True)

        # Store in DB
        today = datetime.today().strftime("%Y-%m-%d")
        conn = sqlite3.connect('review_data.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO reviews (review_text, prediction, confidence, sentiment, date) VALUES (?, ?, ?, ?, ?)", 
                       (review_text, 'FAKE' if prediction else 'GENUINE', confidence, sentiment_label, today))
        conn.commit()
        conn.close()

        # Update wordcloud
        df = pd.read_sql_query("SELECT review_text FROM reviews", sqlite3.connect("review_data.db"))
        generate_wordcloud(df["review_text"].tolist())

# ---------- Sidebar WordCloud ----------
with st.sidebar:
    st.header("📊 Extras")
    if st.button("Show WordCloud", key="show_wc"):
        if os.path.exists("wordcloud.png"):
            st.image("wordcloud.png")
        else:
            st.error("wordcloud.png not found.")
        # ---------- View All Reviews ----------
st.subheader("🗂️ All Reviews")
if st.checkbox("📁 View Stored Reviews"):
    df = pd.read_sql_query("SELECT * FROM reviews", sqlite3.connect("review_data.db"))

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors='coerce')

    # Keyword filter
    search = st.text_input("🔍 Search Review Text")
    if search:
        df = df[df["review_text"].str.contains(search, case=False, na=False)]

    # Sentiment filter
    sentiment_filter = st.selectbox("📎 Sentiment Filter", options=["All", "Positive", "Neutral", "Negative"])
    if sentiment_filter != "All":
        df = df[df["sentiment"] == sentiment_filter]

    # Prediction filter
    prediction_filter = st.selectbox("📎 Prediction Filter", options=["All", "FAKE", "GENUINE"])
    if prediction_filter != "All":
        df = df[df["prediction"] == prediction_filter]

    # Date range filter
    min_date = st.date_input("Start Date", value=df["date"].min().date())
    max_date = st.date_input("End Date", value=df["date"].max().date())
    df = df[(df["date"] >= pd.to_datetime(min_date)) & (df["date"] <= pd.to_datetime(max_date))]

    # Admin tagging
    for i, row in df.iterrows():
        st.markdown(f"**Review #{row['id']}** — *{row['date'].date()}*")
        st.write(row["review_text"])
        st.write(f"Prediction: `{row['prediction']}` | Sentiment: `{row['sentiment']}` | Admin Tag: `{row['admin_tag'] or 'Not tagged'}`")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Mark as Genuine", key=f"genuine_{row['id']}"):
                tag_review(row["id"], "GENUINE")
        with col2:
            if st.button("🚨 Mark as Fake", key=f"fake_{row['id']}"):
                tag_review(row["id"], "FAKE")
        st.markdown("---")

    # Charts
    st.subheader("📈 Summary Charts")
    pie1 = df["prediction"].value_counts().plot.pie(autopct="%.1f%%", title="Prediction Distribution")
    st.pyplot(pie1.figure)

    pie2 = df["sentiment"].value_counts().plot.pie(autopct="%.1f%%", title="Sentiment Distribution")
    st.pyplot(pie2.figure)

    # Export
    st.download_button("📥 Export Filtered Reviews to CSV", data=df.to_csv(index=False), file_name="filtered_reviews.csv", mime="text/csv")
      
        
  
