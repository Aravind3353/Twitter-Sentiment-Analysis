import streamlit as st
import joblib
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

st.title("Twitter Sentiment Analyzer 🐦")
st.subheader("Enter a tweet and see if it's Positive or Negative!")

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

tweet = st.text_area("Enter Tweet")

def clean_tweet(tweet: str) -> str:
    """
    Lowercase, remove URLs/mentions/hashtags/punctuation/numbers,
    then tokenize with RegexpTokenizer and lemmatize.
    """
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\.\S+', '', tweet)
    tweet = re.sub(r'@\w+|#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.strip()

    words = tokenizer.tokenize(tweet)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

if st.button("Analyze Sentiment"):
    if tweet.strip():
        clean = clean_tweet(tweet)
        vec = vectorizer.transform([clean]).toarray()
        
        proba = model.predict_proba(vec)[0][1]

        if proba > 0.6:
            label = "🙂 Positive"
        elif proba < 0.4:
            label = "☹️ Negative"
        else:
            label = "😐 Neutral"

        st.success(f"Sentiment: {label} (score: {proba:.2f})")
    else:
        st.warning("Please enter a tweet first.")
