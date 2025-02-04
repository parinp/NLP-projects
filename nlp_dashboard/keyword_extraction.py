import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    """Extract important keywords using spaCy."""
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop]
    return keywords
