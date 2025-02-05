import streamlit as st
import os
from dotenv import load_dotenv
import json
import requests
from newspaper import Article
import urllib.parse
from transformers import pipeline, BartTokenizer, T5Tokenizer
from duckduckgo_search import DDGS
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk

nltk.download('punkt_tab')

@st.cache_resource(max_entries=1)
def load_summarizer():
    # return pipeline("summarization", model="facebook/bart-large-cnn"), BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6"), BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    # return pipeline("summarization", model="google-t5/t5-small"),   T5Tokenizer.from_pretrained("google-t5/t5-small")

load_dotenv()

MAX_TOKENS = 1022
summarizer, model_tokenizer = load_summarizer()
# API_KEY = os.getenv("NEWSAPI_KEY")
API_KEY = st.secrets["NEWSAPI_KEY"]

# Function to fetch article links from DuckDuckGo Search
def fetch_articles_from_url(query, max_results=10):
    """
    Fetch articles using DuckDuckGo API, prioritizing reputable sources.
    Returns a list of dictionaries containing the URL and title of each article.
    """
    # Reputable domains
    reputable_domains = ["www.bbc.co.uk", "www.bloomberg.com", "edition.cnn.com", "www.cnbc.com", 
                         "www.cnbc.com", "www.aljazeera.com"]
    
    # DuckDuckGo Search
    results = DDGS().news(keywords=query, max_results=(max_results * 5))
    
    # Populating resultant articles
    reputable_articles = []
    other_articles = []
    for result in results:
        url = result['url']
        domain = urllib.parse.urlparse(url).netloc
        
        # Check if the domain is reputable
        if any(reputable_domain in domain for reputable_domain in reputable_domains):
            reputable_articles.append(result)
        else:
            other_articles.append(result)
    
    # Combine articles
    all_articles = reputable_articles + other_articles
    
    # Limit the total number of articles to max_results
    if len(all_articles) > max_results:
        all_articles = all_articles[:max_results]
    
    return [{'url': result['url'], 'title': result['title']} for result in all_articles]

def fetch_articles_from_newsapi(query, max_results=10):
    """
    Fetch articles sorted by relevancy using NewsAPI.org.
    """

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"sortBy=relevancy&"  # Sort by relevancy
        f"pageSize=100&"  
        f"apiKey={API_KEY}"
    )
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])

            # Filter out articles with inaccessible domains
            inaccessible_domains = ["yahoo.com"]
            articles = [
                article for article in articles
                if not any(inaccessible_domain in urllib.parse.urlparse(article['url']).netloc
                          for inaccessible_domain in inaccessible_domains)
            ]

            # Add a 'reputability_score' to each article
            # for article in articles:
            #     domain = urllib.parse.urlparse(article['url']).netloc
            #     article['reputability_score'] = 1 if any(reputable_domain in domain for reputable_domain in reputable_domains) else 0
            
            # Sort articles by reputability, then by published time
            # articles.sort(key=lambda x: (x['reputability_score'], x['publishedAt']), reverse=True)
            # articles.sort(key=lambda x:  x['publishedAt'], reverse=True)
            
            return [
                {
                    "url": article['url'],
                    "title": article['title'],
                    "description": article['description'],
                    "source": article['source']['name']
                }
                for article in articles[:max_results]
            ]
        else:
            print(f"Error fetching articles: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []

# Function to extract key sentences using Sumy
def extract_key_sentences(text, num_sentences=10):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Function to fetch and summarize the article
def fetch_and_summarize(urls):
    all_articles = []
    
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            key_sentences = extract_key_sentences(text)
            all_articles.append(key_sentences)
        
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    return all_articles

# Combine multiple summaries into one
def summarize_articles(articles):
    
    combined_text = " ".join(articles)
    print(f"Intial Tokenized Length: {len(model_tokenizer.tokenize(combined_text))}")

    tokens = model_tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    combined_text = model_tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

    print(f"Final Tokenized Length: {len(model_tokenizer.tokenize(combined_text))}")
    
    # Summarize the combined text
    summary = summarizer(combined_text, max_length=500, min_length=100, do_sample=False)[0]['summary_text']
    
    return summary
