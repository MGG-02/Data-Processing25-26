### --- TF-IDF computes the score for specific terms in Documents. How many times a word appears in the texts. TF = nº_appearances / total_nºwords --- ###

import os
import json
import pandas as pd


from DatasetGeneralDescription import load_pheme_data

base_path = 'pheme-rumour-scheme-dataset'
df = load_pheme_data(base_path)

# Create thread_id -> full text (source + replies) dictionary
thread_texts = {}

threads_dir = os.path.join(base_path, 'threads')
for lang in os.listdir(threads_dir):
    lang_path = os.path.join(threads_dir, lang)
    for event in os.listdir(lang_path):
        event_path = os.path.join(lang_path, event)
        for thread_id in os.listdir(event_path):
            thread_path = os.path.join(event_path, thread_id)
            source_file = os.path.join(thread_path, 'source-tweets', thread_id + '.json')
            replies_dir = os.path.join(thread_path, 'reactions')

            # Read source tweet
            with open(source_file, 'r') as f:
                source_tweet = json.load(f)
                full_text = source_tweet.get('text', '')

            # Append replies
            if os.path.exists(replies_dir):
                for reply_file in os.listdir(replies_dir):
                    with open(os.path.join(replies_dir, reply_file), 'r') as rf:
                        reply = json.load(rf)
                        reply_text = reply.get('text', '')
                        full_text += ' ' + reply_text  # simple space concatenation

            # Store it
            thread_texts[thread_id] = full_text

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert to DataFrame
tfidf_df = pd.DataFrame(list(thread_texts.items()), columns=['thread_id', 'full_text'])

# Initialize vectorizer (you can adjust parameters like stop_words)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(tfidf_df['full_text'])

# Optional: get feature names
feature_names = vectorizer.get_feature_names_out()

# Convert to DataFrame for inspection
tfidf_result = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=tfidf_df['thread_id'])
