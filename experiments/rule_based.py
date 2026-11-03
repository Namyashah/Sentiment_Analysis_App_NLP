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

import pandas as pd
from exploration_preprocessing.data_preprocessing import clean_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score

# Positive vocabulary
positive_words = {
    "good", "great", "excellent", "amazing", "awesome",
    "wonderful", "fantastic", "outstanding", "superb",
    "perfect", "best", "beautiful", "brilliant", "impressive",
    "love", "loved", "like", "liked", "enjoy", "enjoyed",
    "happy", "satisfied", "delighted", "pleased", "excited",
    "recommend", "recommended", "reliable", "durable",
    "comfortable", "smooth", "efficient", "effective",
    "premium", "valuable", "affordable", "worthwhile",
    "quality", "sturdy", "attractive", "convenient",
    "fast", "quick", "easy", "helpful", "friendly",
    "polite", "professional", "responsive", "accurate",
    "successful", "useful", "practical", "beautiful",
    "strong", "soft", "clear", "bright", "fresh",
    "innovative", "exceptional", "remarkable", "fantastic",
    "fantastic", "trustworthy", "excellent", "satisfactory",
    "smoothly", "perfectly", "quickly", "beautifully",
    "improved", "improvement", "benefit", "beneficial",
    "success", "winner", "superior", "convenient",
    "pleasant", "impressive", "enjoyable", "positive"
}

# Negative vocabulary
negative_words = {
    "bad", "poor", "terrible", "awful", "horrible",
    "worst", "disappointing", "disappointed", "useless",
    "waste", "broken", "damaged", "defective", "faulty",
    "cheap", "weak", "fragile", "uncomfortable",
    "unhappy", "unsatisfied", "dissatisfied", "frustrated",
    "annoying", "irritating", "angry", "upset",
    "hate", "hated", "dislike", "disliked", "regret",
    "problem", "problems", "issue", "issues", "failure",
    "failed", "failing", "error", "errors", "crash",
    "crashed", "slow", "sluggish", "difficult",
    "confusing", "complicated", "expensive", "overpriced",
    "unreliable", "inconsistent", "inaccurate",
    "unhelpful", "rude", "late", "delayed", "missing",
    "lost", "dirty", "noisy", "weakness", "inferior",
    "disaster", "disasterous", "unpleasant", "painful",
    "unacceptable", "unsatisfactory", "negative",
    "poorly", "badly", "worst", "failure",
    "worse", "worsened", "damaging", "damages",
    "refund", "complaint", "complaints", "broken",
    "unstable", "inconvenient", "uncomfortable",
    "unfortunately", "defective", "fault", "faulty",
    "disaster", "frustrating", "frustration",
    "disappointing", "drain", "drains", "drained",
    "cheap", "unresolved", "resolve", "worse",
    "poor", "overpriced", "unreliable", "defective",
    "frustrating", "frustrated", "slowly",
    "delay", "delayed", "discomfort"
}

# Negation words
negation_words = {
    "not", "never", "no", "neither", "nor",
    "hardly", "barely", "isnt", "wasnt", "dont",
    "doesnt", "didnt", "cant", "couldnt", "wont"}

# Sentiment prediction function
def rule_based_sentiment(review):
    """
    Predicts positive or negative sentiment using
    predefined vocabulary and basic negation handling.
    """
    words = clean_text(review)

    if isinstance(words, str):
        words = words.split()

    positive_score = 0
    negative_score = 0

    for i, word in enumerate(words):

        # Check whether a sentiment word is negated
        start = max(0, i - 3)

        negated = any(token in negation_words for token in words[start:i])

        if word in positive_words:

            if negated:
                negative_score += 1
            else:
                positive_score += 1

        elif word in negative_words:

            if negated:
                positive_score += 1
            else:
                negative_score += 1

    # Final binary classification
    if positive_score == 0 and negative_score == 0:
        return "neutral"

    elif positive_score > negative_score * 1.5:
        return "positive"

    elif negative_score > positive_score * 1.5:
        return "negative"
    else:
        return "neutral"

    """
        Classifies customer reviews using predefined positive and negative word lists.
        Calculates sentiment scores, handles negation, and predicts positive, negative,
        or neutral sentiment based on the final scores.
    """

if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv(r"C:\Users\namya\Documents\Red and White\NLP\Practical Exam\datasets\customer_reviews_sentiment_practice.csv")

    data = data[data["sentiment"] != "mixed"].copy()  # Low Value Counts Are There And It Creates Complex Patterns Between Classes So We Have Droped It

    # Extract reviews and actual sentiments
    x_test = data["review_text"]
    y_test = data["sentiment"].str.lower()

    # Generate predictions using the rule-based function
    y_pred = x_test.apply(rule_based_sentiment)

    # Compare actual and predicted sentiments
    comparison = pd.DataFrame({
        "Review": x_test,
        "Actual Sentiment": y_test,
        "Predicted Sentiment": y_pred})

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=["positive", "negative", "neutral"],
        zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        y_pred,
        labels=["positive", "negative", "neutral"]))


