import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER if not available
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return "Neutral"  # Handle missing or empty reviews
    
    sentiment_score = sia.polarity_scores(text)["compound"]
    if sentiment_score >= 0.05:
        return "Positive"
    elif sentiment_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Load Data
try:
    df = pd.read_csv("reviews.csv", encoding="ISO-8859-1")
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("Dataset 'reviews.csv' not found! Please upload the correct file.")
    st.stop()

# Show dataset preview
st.title("📊 Customer Review Sentiment Analysis")
st.write("### Data Preview:")
st.write(df.head())

# Check if 'Comments' column exists
if "Comments" not in df.columns:
    st.error("Dataset must contain a 'Comments' column for sentiment analysis.")
    st.stop()

# Apply sentiment analysis
st.write("### Running Sentiment Analysis...")
df["Sentiment"] = df["Comments"].apply(analyze_sentiment)

# Display sample results
st.write(df[["Comments", "Sentiment"]].head())

# Sentiment distribution visualization
sentiment_counts = df["Sentiment"].value_counts()

st.write("### Sentiment Distribution:")
fig, ax = plt.subplots()
colors = {"Positive": "green", "Negative": "red", "Neutral": "blue"}
sentiment_counts.plot(kind='bar', color=[colors[s] for s in sentiment_counts.index])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Customer Sentiment Analysis")
st.pyplot(fig)

# Download cleaned dataset with sentiments
df.to_csv("Sentiment_Analyzed_Reviews.csv", index=False)
st.download_button("Download Analyzed Data", "Sentiment_Analyzed_Reviews.csv")
