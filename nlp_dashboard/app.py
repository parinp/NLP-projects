import streamlit as st
import pandas as pd
import urllib.parse
from wordcloud_visualizer import generate_wordcloud
from sentiment_analysis import analyze_sentiment
from news_scraper import MODELS, load_summarizer, fetch_articles_from_newsapi, fetch_and_summarize, summarize_articles

# Streamlit UI
st.title("Interactive Specific News Summarizer")

# User input for topic
topic = st.text_input("Enter a topic you're interested in:")

# Streamlit dropdown for summarization model selection
selected_model = st.selectbox(
    "Choose a Summarization Model:", 
    list(MODELS.keys())
)

# Load the selected model dynamically
@st.cache_resource(max_entries=1)
def get_summarizer(model_name):
    return load_summarizer(model_name)

summarizer, model_tokenizer = get_summarizer(selected_model)

# Display selected model details
st.write(f"**Selected Model:** {selected_model}")
st.write(f"🔹 **Speed & Accuracy Info:** {selected_model.split('(')[-1].strip(')')}")

if st.button("Get News"):
    if topic.strip():
        articles = fetch_articles_from_newsapi(topic)
        if articles:
            st.write("Fetching news articles...")

            # Create a DataFrame for the articles
            df = pd.DataFrame(articles)

            # Extract the source domain from the URL
            df['source'] = df['url'].apply(lambda x: urllib.parse.urlparse(x).netloc)

            # Format the URL as a clickable hyperlink
            df['url'] = df['url'].apply(lambda x: f'[Link]({x})')

            # Display the DataFrame in a table
            st.write("**Articles:**")
            st.markdown(df[['title', 'source', 'url']].to_markdown(index=False), unsafe_allow_html=True)

            st.write("Summarizing news articles...")

            # Method 1: Fetch using newspaper3k
            articles_url = [article["url"] for article in articles]
            articles_description = fetch_and_summarize(articles_url)

            # Method 2: Fetch using NewsAPI descriptions (Alternative)
            # articles_description = [article["description"] for article in articles]

            # Summarize using the selected model
            combined_summary = summarize_articles(articles_description, summarizer, model_tokenizer)

            # Display combined summary
            st.write("**Combined Summary:**")
            st.write(combined_summary)

            st.write("**WordCloud:**")
            # Generate and display Word Cloud
            wordcloud = generate_wordcloud(combined_summary)
            st.image(wordcloud.to_array())

            st.write("Generating sentiment analysis...")

            # Perform sentiment analysis on the summary
            label, score = analyze_sentiment(combined_summary)
            st.write(f"Sentiment: {label} with confidence {score:.2f}")

        else:
            st.warning("No relevant articles found.")
    else:
        st.warning("Please enter a topic to search.")
