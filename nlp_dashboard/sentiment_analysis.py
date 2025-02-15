import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_sentiment_model():
    # return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

sentiment_analyzer = load_sentiment_model()

# Function to analyze sentiment of a given text
def analyze_sentiment(text):
    """
    Analyze the sentiment of the text and return the label and confidence score.
    The categories are Positive, Negative, and Neutral.
    """
    candidate_labels = ["Positive", "Negative", "Neutral"]
    result = sentiment_analyzer(text, candidate_labels)

    label = result['labels'][0] 
    score = result['scores'][0]
    
    return label, score
