import sys
import pandas as pd
from pathlib import Path
'''
Justification:
    PROJECT_ROOT identifies the main project directory.
    sys.path.insert() adds this directory to Python's module search path,
    allowing us to import our custom data_preprocessing.py file.
    The print statements verify the project path and confirm that
    the required file exists.
'''
# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent

# We Will Start The Vectorization Of Tokens From Importing The Clean_text Function from data_preprocessing file
import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from vectorization.word_2_vec import DocumentVectorizer
from experiments.rule_based import rule_based_sentiment

# Load saved models
MODEL_DIR = PROJECT_ROOT / "models"

bow_model = joblib.load(MODEL_DIR / "bow_model.pkl")
tfidf_model = joblib.load(MODEL_DIR / "tf_idf_model.pkl")
word2vec_model = joblib.load(MODEL_DIR / "word2vec.pkl")

data = pd.read_csv(r"C:\Users\namya\Documents\Red and White\NLP\Practical Exam\datasets\testing_data.csv")

x = data[["review_text"]]
y = data["sentiment"]

# Predictions from both models
bow_predictions = bow_model.predict(x).ravel()
tfidf_predictions = tfidf_model.predict(x).ravel()
word2vec_predictions = word2vec_model.predict(x).ravel()
rule_based_predictions = x["review_text"].apply(rule_based_sentiment)

# Compare results
comparison = pd.DataFrame({
    "Review": x["review_text"],
    "Actual": y,
    "BoW Prediction": bow_predictions,
    "TF-IDF Prediction": tfidf_predictions,
    "Word2Vec Prediction":word2vec_predictions,
    "Rule-Based Prediction": rule_based_predictions})
print(comparison)

comparison["BoW Correct"] = (comparison["Actual"] == comparison["BoW Prediction"])
comparison["TF-IDF Correct"] = (comparison["Actual"] == comparison["TF-IDF Prediction"])
comparison["Word2Vec Correct"] = (comparison["Actual"] == comparison["Word2Vec Prediction"])
comparison["Rule Based Correct"] = (comparison["Actual"] == comparison["Rule-Based Prediction"])

# Calculate correct predictions
bow_correct = comparison["BoW Correct"].sum()
tfidf_correct = comparison["TF-IDF Correct"].sum()
word2vec_correct = comparison["Word2Vec Correct"].sum()
rule_based_correct = comparison["Rule Based Correct"].sum()

print("\n--- Model Comparison ---")
print(f"BoW Correct Predictions: {bow_correct}/{len(data)}")
print(f"TF-IDF Correct Predictions: {tfidf_correct}/{len(data)}")
print(f"Word2Vec Prediction: {word2vec_correct}/{len(data)}")
print(f"Rule Based Prediction: {rule_based_correct}/{len(data)}")

print(f"BoW Accuracy: {bow_correct / len(data) * 100:.2f}%")
print(f"TF-IDF Accuracy: {tfidf_correct / len(data) * 100:.2f}%")
print(f"Word2Vec: {word2vec_correct / len(data) * 100:.2f}%")
print(f"Rule Based: {rule_based_correct / len(data) * 100:.2f}%")

print("\n--- Macro F1 Comparison ---")
print(f"BoW Macro F1: {f1_score(y, bow_predictions, average='macro'):.3f}")
print(f"TF-IDF Macro F1: {f1_score(y, tfidf_predictions, average='macro'):.3f}")
print(f"Word2Vec Macro F1: {f1_score(y, word2vec_predictions, average='macro'):.3f}")
print(f"Rule based Macro F1: {f1_score(y, rule_based_predictions, average='macro'):.3f}")

"""
MODEL COMPARISON AND ANALYSIS

All four sentiment classification models were evaluated using the
same testing dataset containing 60 customer reviews.

CONCLUSION:
    Rule-Based Classification achieved the best performance, correctly
    predicting 58 out of 60 reviews.

    It performed better because it uses a predefined sentiment
    vocabulary, direct word matching, and basic negation handling.
    This allows it to identify positive and negative expressions
    without requiring a large training dataset.

    BoW and TF-IDF depend on patterns learned from training data,
    while Word2Vec uses learned word embeddings. These approaches
    may struggle when the testing reviews contain language patterns
    that differ from their training data.

    Therefore, Rule-Based Classification was the best-performing
    approach in this particular experiment.

NOTE:
    These results are specific to the given testing dataset.
    Rule-based classification may not perform equally well on
    larger, diverse real-world reviews, especially those involving
    sarcasm, context, or complex negation.
"""