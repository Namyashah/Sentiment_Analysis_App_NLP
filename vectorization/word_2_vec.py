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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# We Will Start The Vectorization Of Tokens From Importing The Clean_text Function from data_preprocessing file
import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report,confusion_matrix
from exploration_preprocessing.data_preprocessing import clean_text

data = pd.read_csv(r"C:\Users\namya\Documents\Red and White\NLP\Practical Exam\datasets\customer_reviews_sentiment_practice.csv")
# Remove rows with mixed sentiment
data = data[data["sentiment"] != "mixed"].copy()      # Low Value Counts Are There And It Creates Complex Patterns Between Classes So We Have Droped It

x = data[["review_text"]]
y = data["sentiment"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

class DocumentVectorizer(BaseEstimator, TransformerMixin):
    '''
        Converts each customer review into a fixed-size vector
        by averaging the embeddings of its known words.
    '''
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model

    def fit(self, X, y=None):
        return self

    def transform(self, reviews):
        vectors = []

        for review in reviews:
            tokens = clean_text(review)

            word_vectors = [
                self.word2vec_model.wv[word]
                for word in tokens
                if word in self.word2vec_model.wv]

            if word_vectors:
                mean_vector = np.mean(word_vectors, axis=0)
                max_vector = np.max(word_vectors, axis=0)

                vectors.append(
                    np.concatenate([mean_vector, max_vector]))
            else:
                vectors.append(np.zeros(180))
        return np.array(vectors)

def word2vec(x_train, y_train, x_test, model):
    """
    Trains Word2Vec Skip-gram embeddings and uses a selected
    classifier for customer sentiment classification.
    """
    sentences = x_train["review_text"].apply(clean_text).tolist()

    word2vec_model = Word2Vec(
        sentences=sentences,
        vector_size=75,
        window=3,
        min_count=2,
        workers=4,
        sg=1,
        epochs=50)

    word2vec_preprocessor = ColumnTransformer([
        ("word2vec",DocumentVectorizer(word2vec_model),"review_text")], remainder="drop")

    word2vec_pipeline = Pipeline([
        ("preprocessor", word2vec_preprocessor),
        ("classifier", model)
    ])

    # Train model
    word2vec_pipeline.fit(x_train, y_train)
    y_pred_word2vec = word2vec_pipeline.predict(x_test)

    print("Word2Vec Classification Report:")
    print(classification_report(y_test, y_pred_word2vec))
    print("Word2Vec Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_word2vec))

    # Save model
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "word2vec.pkl"
    joblib.dump(word2vec_pipeline, model_path)

    return word2vec_pipeline, y_pred_word2vec

"""
EXPERIMENT: Word2Vec (Skip-gram)
    Uses the Skip-gram architecture (sg=1) to learn word embeddings.
    Predicts surrounding context words from a target word.
    Skip-gram was selected instead of CBOW to learn word relationships
    by predicting context words from individual target words.

Limitations:
    The small dataset provides limited word-context examples.
    Rare words may not receive meaningful embeddings.
    The model may fail to capture complex semantic relationships.
    Increasing vector dimensions or training epochs does not guarantee
    better results and may lead to overfitting.
"""

'''
    The `if __name__ == "__main__":` block ensures that dataset loading,
    testing, and printing run only when this file is executed directly.
'''
if __name__ == "__main__":
    word2vec_pipeline, y_pred_word2vec = word2vec(x_train,y_train,x_test,LogisticRegression())
# UnCommet The Above Code When You Have To Retrain The Model