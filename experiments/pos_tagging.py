import sys
from pathlib import Path
'''
Justification:
    PROJECT_ROOT identifies the main project directory.
    sys.path.insert() adds this directory to Python's module search path,
    allowing us to import our custom data_preprocessing.py file.
'''
# Get the main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project directory to Python path
sys.path.insert(0, str(PROJECT_ROOT))

import spacy
import pandas as pd

data = pd.read_csv(r"C:\Users\namya\Documents\Red and White\NLP\Practical Exam\datasets\customer_reviews_sentiment_practice.csv")
nlp = spacy.load("en_core_web_sm")

"""
EXPERIMENT: POS (Part-of-Speech) Tagging
    Uses spaCy to identify grammatical categories such as nouns,
    verbs, adjectives and adverbs in customer reviews.
    Helps identify frequently discussed product features and
    customer opinions.
    Used for linguistic analysis and customer insights.

Limitations:
    May produce incorrect tags for short, informal, misspelled
    or domain-specific reviews.
    We Have Not Used POS Tagging Because It Can't Be Used With The Clean_text() Function Properly.
"""
def pos_filter(review):
    """
    Retains nouns, verbs, adjectives, adverbs and negation words.
    """
    doc = nlp(review)

    allowed_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    negation_words = {"not", "never", "no", "nor", "neither"}

    filtered_words = [
        token.text
        for token in doc
        if token.pos_ in allowed_pos or token.text.lower() in negation_words]

    return " ".join(filtered_words)

'''
    The `if __name__ == "__main__":` block ensures that dataset loading,
    testing, and printing run only when this file is executed directly.
    When importing `clean_text()` into another file, these statements
    are skipped, preventing unnecessary execution.
'''

if __name__ == "__main__":

    sample = "The product is extremely good but the delivery was not satisfactory."

    result = pos_filter(sample)

    print("Original Text:", sample)
    print("After POS Filtering:", result)