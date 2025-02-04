import requests
from newspaper import Article
import urllib.parse
from transformers import pipeline
from transformers import BartTokenizer
from duckduckgo_search import DDGS

# Load pre-trained model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Function to create the correct search URL
def encode_query(query):
    query_encoded = urllib.parse.quote_plus(query)
    return query_encoded

# Function to fetch article links from the search result page
def fetch_articles_from_url(query, max_results=10, max_pages=10):
    """
    Fetch articles using DuckDuckGo API
    """
    
    # query_encoded = urllib.parse.quote_plus(query)
    results = DDGS().news(keywords = query, 
                          max_results=max_results)

    article_links = []

    # Collect articles from preferred sources
    for result in results:
        url = result['url']
        article_links.append(url)

    return article_links

# Function to fetch and summarize the article
def fetch_and_summarize(urls):
    all_articles = []
    
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            
            all_articles.append({'url': url, 'text': text})
        
        except Exception as e:
            print(f"⚠️ Error processing {url}: {e}")
    
    return all_articles

# Combine multiple summaries into one
def summarize_articles(articles):
    combined_text = ""
    for article in articles:
        combined_text += article['text'] + " "
    
    # print(combined_text)
    if len(combined_text) > 200:
        tokens = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=1000)
        combined_text = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
    
    print(f"🔹 Final Tokenized Length: {len(tokenizer.tokenize(combined_text))}")
    
    # Summarize the combined text
    summary = summarizer(combined_text, max_length=300)[0]['summary_text']
    
    return summary
