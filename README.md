![Plots Screenshot](images/Screenshot%202026-11-03%20133158.png)
# 💬 Customer Sentiment Analysis Using NLP

> An end-to-end NLP project that analyzes customer reviews and
> classifies them as **Positive**, **Negative**, or **Neutral**. It
> includes text preprocessing, NLP experiments, model comparison, and an
> interactive Streamlit application.

## ✨ Project Overview

The goal of this project is to understand customer opinions from written
reviews and automatically identify their sentiment.

The project explores multiple NLP techniques and compares their
performance. A rule-based classifier was selected for the final
Streamlit application because it achieved the strongest result in the
project's 60-review comparison test.

## 🚀 Features

-   🧹 Text preprocessing: HTML and URL removal, lowercase conversion,
    contraction handling, tokenization, and lemmatization.
-   🏷️ POS tagging: retains nouns, verbs, adjectives, and adverbs while
    preserving important negation words.
-   🔢 Text representation experiments: Bag-of-Words, TF-IDF, and
    Word2Vec.
-   🧠 Sentiment classification: machine-learning pipelines and a
    vocabulary-based rule classifier.
-   📊 Evaluation: accuracy, macro F1-score, classification report, and
    confusion matrix.
-   🌐 Streamlit application:
    -   Analyze one review through manual text input.
    -   Upload a CSV containing a `review_text` column.
    -   View predictions and sentiment counts.
    -   Download the predicted results as a CSV file.

## 🧰 Technologies Used

  Technology      Purpose
  --------------- -----------------------------------------------
  Python          Main programming language
  Pandas, NumPy   Data handling and numerical operations
  NLTK            Tokenization and lemmatization
  spaCy           Part-of-speech tagging
  Scikit-learn    Vectorization, classification, and evaluation
  Streamlit       Interactive web application

## 🗂️ Project Structure

``` text
Practical Exam/
│
├── datasets/
│   └── customer_reviews_sentiment_practice.csv
│
├── exploration_preprocessing/
│   └── data_preprocessing.py
│
├── experiments/
│   ├── pos_tagging.py
│   └── rule_based.py
│
├── vectorization/
│   ├── bag_of_words.py
│   ├── tf_idf.py
│   └── word_2_vec.py
│
├── models/
│   ├── bow_model.pkl
│   ├── tf_idf_model.pkl
│   └── word2vec.pkl
│
├── main.py
├── comparison_testing_data.py
└── requirements.txt
```

*Your local folder may contain additional files, notebooks, or test
datasets depending on your experiments.*

## ⚙️ Installation & Setup

### 1. Clone or download the project

Open a terminal in the project directory.

### 2. Create and activate a virtual environment (recommended)

``` bash
python -m venv venv
```

**Windows:**

``` bash
venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Download the spaCy English language model

``` bash
python -m spacy download en_core_web_sm
```

If your NLTK setup requires the WordNet resources used by lemmatization,
run:

``` python
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## ▶️ Execution

Run the Streamlit application from the main project directory:

``` bash
streamlit run main.py
```

Streamlit will open the application in your browser. If it does not open
automatically, use the local URL shown in the terminal.

To run the model experiments or comparison script, execute the relevant
Python file from the project root, for example:

``` bash
python comparison_testing_data.py
```

## 🖥️ How to Use the Application

### ✍️ Single Review Analysis

1.  Open the **Single Review Analysis** tab.
2.  Enter a customer review in the text area.
3.  Click **Analyze Sentiment**.
4.  View the predicted label: Positive, Negative, or Neutral.

### 📁 CSV Review Analysis

1.  Prepare a CSV file with a column named `review_text`.
2.  Open the **CSV Review Analysis** tab.
3.  Upload the CSV file.
4.  Click **Analyze All Reviews**.
5.  View each review with its `predicted_sentiment` and the sentiment
    counts.
6.  Download the prediction results as a CSV file.

**Example input CSV:**

  review_text
  --------------------------------------------------------------
  The product quality is excellent and delivery was quick.
  The item stopped working and customer support was unhelpful.
  The package arrived yesterday with the instruction manual.

## 🔬 Project Workflow

1.  **Data understanding:** inspected customer review data and sentiment
    labels.
2.  **Preprocessing:** removed HTML tags and URLs, expanded common
    contractions to preserve negation, normalized text, tokenized, and
    lemmatized.
3.  **POS filtering:** used spaCy to retain nouns, verbs, adjectives,
    and adverbs, while preserving negation terms such as `not`, `never`,
    and `no`.
4.  **Vectorization experiments:** implemented BoW, TF-IDF, and Word2Vec
    representations.
5.  **Classification experiments:** trained and evaluated
    machine-learning classifiers and developed a rule-based sentiment
    baseline using positive/negative vocabularies and basic negation
    handling.
6.  **Model comparison:** compared predictions using accuracy and macro
    F1-score on a separate 60-review test set.
7.  **Application development:** integrated the rule-based classifier
    into Streamlit for single-review and CSV-based predictions.

## 📊 Model Comparison

Reported results from the project's 60-review comparison test:

  Approach           Accuracy    Macro F1
  -------------- ------------ -----------
  Bag-of-Words         46.67%       0.444
  TF-IDF               48.33%       0.483
  Word2Vec             33.33%       0.195
  Rule-Based       **96.67%**   **0.967**

The rule-based approach performed best on this particular test set,
correctly classifying 58 of 60 reviews. It uses predefined sentiment
vocabulary and basic negation handling, which matched the language in
this test set well.

> **Note:** These scores are specific to the test data used in this
> project. They do not guarantee the same performance on new or
> real-world reviews. Rule-based methods can struggle with sarcasm,
> context, ambiguous wording, and complex negation. Evaluate on
> representative unseen data before relying on predictions.

## 📄 Dataset Format

For CSV prediction, the application requires:

-   A column named `review_text`.
-   One review per row.
-   An optional existing sentiment column may be present; the
    application adds `predicted_sentiment` and does not use the actual
    label to make predictions.

## ⚠️ Important Notes

-   Run commands from the main project directory so the package imports
    resolve correctly.
-   Keep the `exploration_preprocessing`, `experiments`, and other
    Python modules available; the rule-based classifier imports the
    shared preprocessing function.
-   If you change preprocessing or POS filtering, rerun your experiments
    and retrain saved machine-learning pipelines so training and
    inference use the same processing steps.
-   The final Streamlit app uses the rule-based classifier; the other
    approaches are retained for experimentation and comparison.

## 🌟 Future Improvements

-   Expand and refine the sentiment vocabulary.
-   Improve negation and context handling.
-   Add more diverse, independently labelled test reviews.
-   Tune the neutral-class decision rule using validation data.
-   Explore more advanced NLP models and compare them on the same test
    split.

------------------------------------------------------------------------

**Made with Python, NLP, and Streamlit. 💙**