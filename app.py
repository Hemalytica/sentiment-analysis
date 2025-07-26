import streamlit as st

import pandas as pd
 
import matplotlib.pyplot as plt
 

# Load Data
 

df = pd.read_csv("Amazon_Reviews.csv")  # Replace with your dataset
 

# Title
 

st.title("Customer Review Sentiment Analysis")
 

# Show dataset preview
 

st.write("### Data Preview:")
 

st.write(df.head())
 

# Simple sentiment counting (modify based on your logic)
 

sentiment_counts = df['Sentiment'].value_counts()
 

# Visualization
 

st.write("### Sentiment Distribution:")
 

fig, ax = plt.subplots()
 

sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
 

plt.xlabel("Sentiment")
 

plt.ylabel("Count")
 

plt.title("Customer Sentiment Analysis")
 

st.pyplot(fig) 
