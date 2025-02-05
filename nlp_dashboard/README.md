# Specific News Fetcher and Summarizer

This Python script fetches news articles using the NewsAPI.org API primarliy or DuckDuckGo API and prioritizes articles from reputable sources. It returns both the title and URL of each article. This is then used to generate a summary of the news articles using the NLTK library and the TextRank algorithm. The summary is then generated after being passed to a pre-trained model.

## Features
- Fetches articles based on user queries.
- Prioritizes articles from reputable domains (e.g., BBC, Bloomberg, CNN).
- The NewsAPI has a limit of 100 requests per day. After exceeding, it will use DuckDuckGo Search instead.
- Returns structured data (title and URL) for easy use.

## Testing
Please feel free to test the script with your own queries. The script is designed to be flexible.
https://nlp-projects-hhhqvxvtzqngdztqz8vk8p.streamlit.app/