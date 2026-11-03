import sys
from pathlib import Path

# Get the main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))

'''
- We Will Use This data_preprocessing.py file Into Our main.ipynb file by importing Important Functions And According To Our Needs.
'''
# Importing Important Libraries
import re
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.tokenize import wordpunct_tokenize
from experiments.pos_tagging import pos_filter
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download these resources once if they aren't installed
nltk.download("wordnet")
nltk.download("omw-1.4")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Creating A Text Clearning Function
'''
    Preprocess customer review text for NLP and sentiment analysis.

    This function removes HTML tags and URLs, converts text to lowercase,
    removes unwanted symbols, and normalizes extra whitespace. It preserves
    important negation words such as "not" and "never", as these can change
    the sentiment of a review. Common negation contractions are expanded
    where specified.

    The cleaned text is then tokenized into individual words, stemmed to
    reduce words to approximate root forms, and lemmatized to obtain their
    dictionary base forms.
'''
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Convert common contractions to preserve negation
    text = re.sub(r"\bisn't\b", "is not", text, flags=re.I)
    text = re.sub(r"\baren't\b", "are not", text, flags=re.I)
    text = re.sub(r"\bwasn't\b", "was not", text, flags=re.I)
    text = re.sub(r"\bweren't\b", "were not", text, flags=re.I)
    text = re.sub(r"\bdon't\b", "do not", text, flags=re.I)
    text = re.sub(r"\bdoesn't\b", "does not", text, flags=re.I)
    text = re.sub(r"\bdidn't\b", "did not", text, flags=re.I)
    text = re.sub(r"\bcan't\b", "cannot", text, flags=re.I)
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.I)

    # Convert text to lowercase
    text = text.lower()

    # Keep letters, numbers, and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = wordpunct_tokenize(text)

    # Stemming
    # tokens = [stemmer.stem(word) for word in tokens]   # Use It When There Is A Bigger Number Of Dataset

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # POS Filtering
    text = " ".join(tokens)
    tokens = pos_filter(text).split()

    return tokens

'''
Justification:
    The `if __name__ == "__main__":` block ensures that dataset loading,
    testing, and printing run only when this file is executed directly.
    When importing `clean_text()` into another file, these statements
    are skipped, preventing unnecessary execution.
'''
if __name__ == "__main__":

    # Loading The Dataset Using Pathlib
    DATA_PATH = Path(r"C:\Users\namya\Documents\Red and White\NLP\Practical Exam\datasets\customer_reviews_sentiment_practice.csv")
    df = pd.read_csv(DATA_PATH)

    # Keep the original data unchanged
    data = df.copy()

    # We Have Droped Mixed Class Rows Because It has Low Balancing So We Have Droped that Class Rows
    data = data[data["sentiment"] != "mixed"].copy()

    print("Dataset loaded successfully!")
    print("Shape:", data.shape)
    print(data.head())

    # Applying The clean_text() Function
    '''
        We Can See And Compare That All The Words have Been Lower Cased And Unneccesary fullstops, comas are been removed from the original text and words are back to their root form and base dictionary form with the help of stemming and Lemmatization and Before that converted Text into Tokens Breaking Sentences Into Words.

        And We Have Preserved the original text rather then replacing it with the cleaned_text.
    '''
    data["cleaned_text"] = data["review_text"].apply(clean_text)
    for i in range(5):
        print("Original:", data["review_text"].iloc[i])
        print("Cleaned: ", data["cleaned_text"].iloc[i])
        print("-" * 80)

    # Validating The Dataset 
    required_columns = [
        "review_text",
        "sentiment",
        "review_date",
        "rating",
        "product_category",
        "product_name"]

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    print("Sentiment labels:", sorted(data["sentiment"].dropna().unique()))
    print("Missing review texts:", data["review_text"].isna().sum())