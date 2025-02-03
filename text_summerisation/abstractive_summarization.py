from transformers import pipeline

# Load the T5 summarization model
summarizer = pipeline("summarization", model="t5-small")

def abstractive_summary(text, max_length=100, min_length=30):
    """Generate an abstractive summary using the T5 model."""
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

if __name__ == "__main__":
    sample_text = """Natural Language Processing (NLP) is a field of AI that focuses on the interaction between 
    computers and humans using language. It helps machines understand, interpret, and generate human language effectively. 
    Applications include chatbots, sentiment analysis, machine translation, and more. NLP techniques are widely used in 
    healthcare, finance, and customer service industries to extract insights from unstructured text data."""

    summary = abstractive_summary(sample_text, max_length=50, min_length=20)
    print("Original Text:\n", sample_text)
    print("\nAbstractive Summary:\n", summary)
