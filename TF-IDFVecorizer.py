"""
Full TF-IDF text processing pipeline to generate a CSV file where:

- Each row = one document
- Each column = a word in the vocabulary
- Each cell = TF-IDF weight of that word in the document

Output:
    tfidf_matrix.csv
"""

import os
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------------------------------------
# (1) TEXT CLEANING FUNCTION
# ----------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)

    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----------------------------------------------------------
# (2) LOAD DOCUMENTS (YOU CAN EDIT THIS PART)
# ----------------------------------------------------------

# Example: documents from a dictionary (user provided earlier)
# Replace with your own: thread_texts, source tweets, etc.
def load_documents(base_path="./all-rnr-annotated-threads"):
    documents = []
    doc_ids = []

    for event in os.listdir(base_path):
        event_path = os.path.join(base_path, event)
        if not os.path.isdir(event_path):
            continue

        # Only inside "rumours"
        rumours_path = os.path.join(event_path, "rumours")
        if not os.path.isdir(rumours_path):
            continue

        for thread_id in os.listdir(rumours_path):
            if "_" in thread_id:
                continue

            thread_dir = os.path.join(rumours_path, thread_id)
            source_file = os.path.join(thread_dir, "source-tweets", f"{thread_id}.json")

            if not os.path.exists(source_file):
                continue

            # Read source tweet
            with open(source_file, "r") as f:
                src = json.load(f)
                text = src.get("text", "")

            # Clean the text
            text_clean = clean_text(text)

            # Save
            documents.append(text_clean)
            doc_ids.append(thread_id)

    print(f"Loaded {len(documents)} documents.")
    return documents, doc_ids


# ----------------------------------------------------------
# (3) BUILD TF-IDF MATRIX
# ----------------------------------------------------------
def build_tfidf_matrix(documents, max_features=5000):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2)     # unigrams + bigrams
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names


# ----------------------------------------------------------
# (4) CONVERT TO DENSE CSV (SAFE!)
# ----------------------------------------------------------
def save_tfidf_to_csv(tfidf_matrix, feature_names, doc_ids, output_file="tfidf_matrix.csv"):

    # Convert sparse → dense safely
    tfidf_dense = tfidf_matrix.toarray()

    df = pd.DataFrame(tfidf_dense, index=doc_ids, columns=feature_names)

    df.to_csv(output_file)
    print(f"TF-IDF CSV saved: {output_file}")
    print(f"Shape: {df.shape}")


# ----------------------------------------------------------
# (5) MAIN EXECUTION
# ----------------------------------------------------------
if __name__ == "__main__":

    # Step 1: Load documents
    documents, doc_ids = load_documents()

    # Step 2: Build TF-IDF
    tfidf_matrix, feature_names = build_tfidf_matrix(documents)

    # Step 3: Save to CSV
    save_tfidf_to_csv(tfidf_matrix, feature_names, doc_ids)

    print("Done.")
