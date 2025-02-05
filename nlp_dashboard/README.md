# Specific News Fetcher and Summarizer

This Python script fetches news articles using the NewsAPI.org API primarliy or DuckDuckGo API and prioritizes articles from reputable sources. It returns both the title and URL of each article. This is then used to generate a summary of the news articles using the NLTK library and the TextRank algorithm. The summary is then generated after being passed to a pre-trained model.

## Features
- Fetches articles based on user queries.
- Prioritizes articles from reputable domains (e.g., BBC, Bloomberg, CNN).
- Returns structured data (title and URL) for easy use.