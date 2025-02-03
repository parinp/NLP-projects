from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # Latent Semantic Analysis Summarizer

def extractive_summary(text, num_sentences=3):
    """Generate an extractive summary using the LSA method."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

if __name__ == "__main__":
    sample_text = """Natural Language Processing (NLP) is a field of AI that focuses on the interaction between 
    computers and humans using language. It helps machines understand, interpret, and generate human language effectively. 
    Applications include chatbots, sentiment analysis, machine translation, and more. NLP techniques are widely used in 
    healthcare, finance, and customer service industries to extract insights from unstructured text data."""
    
    summary = extractive_summary(sample_text, num_sentences=2)
    print("Original Text:\n", sample_text)
    print("\nExtractive Summary:\n", summary)
