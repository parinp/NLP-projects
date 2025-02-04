import streamlit as st
from transformers import pipeline
from wordcloud_visualizer import generate_wordcloud
from sentiment_analysis import analyze_sentiment
from news_scraper import fetch_articles_from_url, fetch_and_summarize, summarize_articles, encode_query

# Streamlit UI
st.title("Interactive Specific News Summarizer")

topic = st.text_input("Enter a topic you're interested in:")

if st.button("Get News"):
    if topic.strip():
        # Fetch URLs from BBC and CNN
        search_url = encode_query(topic)
        urls = fetch_articles_from_url(search_url)
        
        if urls:
            st.write("Fetching and summarizing news articles...")
            
            # Fetch and summarize articles
            articles = fetch_and_summarize(urls)
            
            # Combine summaries into one
            combined_summary = summarize_articles(articles)
            
            # Display combined summary
            st.write("**Combined Summary:**")
            st.write(combined_summary)
            
            # Word Cloud and Sentiment Analysis for the combined text
            wordcloud = generate_wordcloud(combined_summary)
            st.image(wordcloud.to_array())

            st.write("Generating sentiment analysis...")

            # Sentiment analysis of the combined text
            label, score = analyze_sentiment(combined_summary)
            st.write(f"Sentiment: {label} with confidence {score:.2f}")

        else:
            st.warning("No relevant articles found.")
    else:
        st.warning("Please enter a topic to search.")
