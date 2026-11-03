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
import pandas as pd
from sklearn.pipeline import Pipeline
from exploration_preprocessing.data_preprocessing import clean_text
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

data = pd.read_csv(r"C:\Users\namya\Documents\Red and White\NLP\Practical Exam\datasets\customer_reviews_sentiment_practice.csv")
# Remove rows with mixed sentiment
data = data[data["sentiment"] != "mixed"].copy()      # Low Value Counts Are There And It Creates Complex Patterns Between Classes So We Have Droped It

x = data[["review_text"]]
y = data["sentiment"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

def tf_idf(x_train,y_train,x_test,model):
    '''
        TF-IDF : Term Frequency-Inverse Document Frequency is a statistical weight used in natural language processing (NLP) and information retrieval to measure how important a word is to a document within a larger collection
    '''
    tf_idf_preprocessor = ColumnTransformer([
        ("tf_idf",TfidfVectorizer(tokenizer=clean_text,token_pattern=None,lowercase=False,ngram_range=(1, 2),max_features=100),"review_text")
    ],remainder="drop")

    tf_idf_pipeline = Pipeline([
        ("preprocessor",tf_idf_preprocessor),
        ("classifier_tf_idf",model)
    ])

    tf_idf_pipeline.fit(x_train,y_train)
    y_pred_tf_idf = tf_idf_pipeline.predict(x_test)
    print(f"\nTF-IDF : \n{classification_report(y_test,y_pred_tf_idf)}")
    print("TF-IDF Confusion Matrix:")
    print(f"{confusion_matrix(y_test, y_pred_tf_idf)}\n")

    # Save trained model
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "tf_idf_model.pkl"
    joblib.dump(tf_idf_pipeline, model_path)

    return tf_idf_pipeline,y_pred_tf_idf

"""
EXPERIMENT 2: TF-IDF
    - TF-IDF converts text into numerical features based on word importance rather than raw frequency.
    - Multinomial Naive Bayes was used as the classifier to maintain a fair comparison with BoW.
    - Common words receive lower importance, while more informative words can receive higher weights.
    - Unigrams and different N-gram configurations were experimented with to capture individual words and meaningful phrases.

Observations:
    - TF-IDF achieved 97% accuracy and 0.96 macro F1-score on the original test set.
    - BoW achieved 96% accuracy and 0.91 macro F1-score.
    - Both models achieved 58.33% accuracy on our 36 manually labelled custom reviews.
    - The custom test showed that both models can struggle with negation, mixed opinions and unseen expressions.

Limitations:
    - Does not understand sentence meaning or deep context.
    - Struggles with sarcasm and complex negation.
    - Cannot capture semantic relationships between words.

Conclusion:
    TF-IDF performed better than BoW on the original test set,
    particularly in macro F1-score. However, it does not guarantee
    better predictions for every individual review.
"""

'''
    The `if __name__ == "__main__":` block ensures that dataset loading,
    testing, and printing run only when this file is executed directly.
'''
if __name__ == "__main__":
    tf_idf_pipeline,y_pred_tf_idf = tf_idf(x_train,y_train,x_test,MultinomialNB())
# UnCommet The Above Code When You Have To Retrain The Model