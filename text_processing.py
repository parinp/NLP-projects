import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords if not already downloaded
nltk.download("stopwords")
nltk.download("punkt_tab")

# Load Spacy model for lemmatization
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Basic text cleaning: lowercasing, punctuation removal, tokenization, stopword removal, and lemmatization."""
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters, numbers, and extra spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization using Spacy
    doc = nlp(" ".join(filtered_tokens))
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    return lemmatized_text

if __name__ == "__main__":
    sample_text = """Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans using language.
                     It helps machines understand, interpret, and generate human language effectively."""
    
    processed_text = clean_text(sample_text)
    print("Original Text:\n", sample_text)
    print("\nProcessed Text:\n", processed_text)
