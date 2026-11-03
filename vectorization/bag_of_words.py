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

def bag_of_words(x_train,y_train,x_test,model):
    '''
    Bag Of Words : Converts Tokens Into Vectors using Counts Based On The Number Of Vocabulary
    '''
    preprocessor = ColumnTransformer([
        ("bow",CountVectorizer(tokenizer=clean_text,token_pattern=None,lowercase=False,ngram_range=(1, 2),max_features=100),"review_text")
    ],remainder="drop")

    bow_pipeline = Pipeline([
        ("preprocessor",preprocessor),
        ("classifier_bow",model)
    ])

    bow_pipeline.fit(x_train,y_train)
    y_pred_bow = bow_pipeline.predict(x_test)
    print(f"Bag Of Words : {classification_report(y_test,y_pred_bow)}")
    print("BoW Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_bow))

    # Save trained model
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "bow_model.pkl"
    joblib.dump(bow_pipeline, model_path)

    return bow_pipeline,y_pred_bow

'''
EXPERIMENT 1: BAG OF WORDS (BoW)
    - Bag of Words converts text into numerical features based on word frequency.
    - Multinomial Naive Bayes was selected because it works well with word-count features and is efficient for  text classification.
    - Different N-gram configurations (unigrams and bigrams) were tested, but predictions remained unchanged.
    - Bigrams capture consecutive word pairs such as "not good" and "very poor", helping identify short phrases.
    - We selected bigrams because they provide additional phrase-level information.

Limitations:
    - Does not understand sentence context or deeper meaning like "not good", "low quality".
    - Struggles with sarcasm, negation and mixed opinions.
    - Ignores long-range word relationships.
    - Can create large, sparse feature matrices.

Conclusion:
    BoW provides a simple and effective baseline for sentiment classification,
    but its limited understanding of context affects prediction quality.
'''

'''
    The `if __name__ == "__main__":` block ensures that dataset loading,
    testing, and printing run only when this file is executed directly.
'''
if __name__ == "__main__":
    bow_pipeline,y_pred_bow = bag_of_words(x_train,y_train,x_test,MultinomialNB())
# UnCommet The Above Code When You Have To Retrain The Model