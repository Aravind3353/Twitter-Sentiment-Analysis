import streamlit as st
import joblib
import re
import string
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np

# Load model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Stopwords and tokenizer
stop_words = set(ENGLISH_STOP_WORDS)
tokenizer = RegexpTokenizer(r'\w+')

# Streamlit UI
st.title("Twitter Sentiment Analyzer 🐦")
st.subheader("Enter a tweet and see if it's Positive or Negative!")
tweet = st.text_area("Enter Tweet")

# Clean function without WordNetLemmatizer
def clean_tweet(tweet: str) -> str:
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\.\S+', '', tweet)
    tweet = re.sub(r'@\w+|#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.strip()
    words = tokenizer.tokenize(tweet)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Prediction
if st.button("Analyze Sentiment"):
    if tweet.strip():
        clean = clean_tweet(tweet)
        vec = vectorizer.transform([clean]).toarray()
        prediction = model.predict(vec)[0]
        label = "🙂 Positive" if prediction == 1 else "☹️ Negative"
        st.success(f"Sentiment: {label}")
    else:
        st.warning("Please enter a tweet first.")
