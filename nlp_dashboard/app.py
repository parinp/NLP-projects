import streamlit as st
import pandas as pd
import urllib.parse
from wordcloud_visualizer import generate_wordcloud
from sentiment_analysis import analyze_sentiment
from news_scraper import fetch_articles_from_newsapi, fetch_articles_from_url, fetch_and_summarize, summarize_articles

# Header UI
st.title("Interactive Specific News Summarizer")

topic = st.text_input("Enter a topic you're interested in:")

if st.button("Get News"):
    if topic.strip():
        articles = fetch_articles_from_newsapi(topic)
        if not articles:
            articles = fetch_articles_from_url(topic)
        if articles:
            st.write("Fetching news articles...")
            
            # Display the articles in a table
            df = pd.DataFrame(articles)
            df['source'] = df['url'].apply(lambda x: urllib.parse.urlparse(x).netloc)
            df['url'] = df['url'].apply(lambda x: f'[Link]({x})')
            st.write("**Articles:**")
            st.markdown(df[['title', 'source', 'url']].to_markdown(index=False), unsafe_allow_html=True)

            st.write("Summarizing news articles...")
            
            # Method 1: Fetch using newspaper4k
            articles_url = [article["url"] for article in articles]
            articles_description = fetch_and_summarize(articles_url)
            
            # Method 2: Fetch using newsapi description
            # articles_description = [article["description"] for article in articles]

            # Combine summaries into one
            combined_summary = summarize_articles(articles_description)
            st.write("**Combined Summary:**")
            st.write(combined_summary)
            
            st.write("**WordCloud:**")
            wordcloud = generate_wordcloud(combined_summary)
            st.image(wordcloud.to_array())

            st.write("Generating sentiment analysis...")
            label, score = analyze_sentiment(combined_summary)
            st.write(f"Sentiment: {label} with confidence {score:.2f}")

        else:
            st.warning("No relevant articles found.")
    else:
        st.warning("Please enter a topic to search.")