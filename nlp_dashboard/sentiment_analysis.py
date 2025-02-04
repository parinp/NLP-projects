from transformers import pipeline

# Load pre-trained model for zero-shot classification
sentiment_analyzer = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to analyze sentiment of a given text
def analyze_sentiment(text):
    """
    Analyze the sentiment of the text and return the label and confidence score.
    The categories are Positive, Negative, and Neutral.
    """
    candidate_labels = ["Positive", "Negative", "Neutral"]  # Define sentiment labels

    # Perform zero-shot classification
    result = sentiment_analyzer(text, candidate_labels)

    # Extract the best label and its score
    label = result['labels'][0]  # The most likely sentiment label
    score = result['scores'][0]  # Confidence score for the label
    
    return label, score
