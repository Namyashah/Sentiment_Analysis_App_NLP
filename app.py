
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Project path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import rule-based model
from experiments.rule_based import rule_based_sentiment


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)


# ---------------- APPLICATION HEADER ----------------

st.title("💬 Customer Sentiment Analysis")
st.write(
    "Analyze customer reviews using NLP and "
    "Rule-Based Sentiment Classification."
)

st.divider()


# ---------------- SENTIMENT DISPLAY ----------------

def display_sentiment(sentiment):

    if sentiment == "positive":
        st.success("😊 Positive Sentiment")

    elif sentiment == "negative":
        st.error("😞 Negative Sentiment")

    else:
        st.info("😐 Neutral Sentiment")


# ---------------- APPLICATION TABS ----------------

single_tab, csv_tab = st.tabs([
    "✍️ Single Review Analysis",
    "📁 CSV Review Analysis"
])


# =====================================================
# SINGLE REVIEW ANALYSIS
# =====================================================

with single_tab:

    st.subheader("Analyze a Single Customer Review")

    review = st.text_area(
        "Enter your customer review:",
        placeholder="Write your product or service experience here...",
        height=150
    )

    if st.button("Analyze Sentiment", key="single"):

        if review.strip():

            prediction = rule_based_sentiment(review)

            st.subheader("Prediction Result")

            display_sentiment(prediction)

            st.write("**Your Review:**")
            st.write(review)

        else:
            st.warning("Please enter a review first.")


# =====================================================
# CSV FILE ANALYSIS
# =====================================================

with csv_tab:

    st.subheader("Analyze Multiple Customer Reviews")

    st.write(
        "Upload a CSV file containing a column named "
        "`review_text`."
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)

            if "review_text" not in data.columns:
                st.error(
                    "CSV must contain a 'review_text' column."
                )

            else:

                st.write("### Uploaded Dataset")
                st.dataframe(data.head())

                if st.button("Analyze All Reviews"):

                    results = data.copy()

                    results["predicted_sentiment"] = (
                        results["review_text"]
                        .fillna("")
                        .astype(str)
                        .apply(rule_based_sentiment)
                    )

                    st.write("### Sentiment Analysis Results")

                    # Display sentiment counts
                    counts = (
                        results["predicted_sentiment"]
                        .value_counts()
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "😊 Positive",
                        counts.get("positive", 0)
                    )

                    col2.metric(
                        "😞 Negative",
                        counts.get("negative", 0)
                    )

                    col3.metric(
                        "😐 Neutral",
                        counts.get("neutral", 0)
                    )

                    # Display results
                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    # Download results
                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=csv_data,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Customer Sentiment Analysis | "
    "NLP Project | Rule-Based Classification"
)
