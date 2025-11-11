import os
import pandas as pd
import json
# Adjust this path to where you've extracted the PHEME dataset
data_path = './pheme-rumour-scheme-dataset'  # Change as needed
def load_pheme_data(data_path):
    all_data = []
    for event in os.listdir(data_path):
        event_path = os.path.join(data_path, event)
        if not os.path.isdir(event_path):
            continue
        for thread in os.listdir(event_path):
            thread_path = os.path.join(event_path, thread, 'source-tweet', thread + '.json')
            if os.path.exists(thread_path):
                with open(thread_path, 'r', encoding='utf-8') as f:
                    tweet = json.load(f)
                tweet['event'] = event
                tweet['thread_id'] = thread
                all_data.append(tweet)
    df = pd.DataFrame(all_data)
    return df
# Load dataset
df = load_pheme_data(data_path)
# Show general description
print("Number of instances (tweets):", df.shape[0])
print("Variables (columns):", list(df.columns))
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())
df.head()
import os
import json
import pandas as pd
# Adjust this path to where you've extracted the PHEME dataset
data_path = './pheme-rumour-scheme-dataset'  # Change as needed
def load_pheme_data(base_path: str) -> pd.DataFrame:
    """Load PHEME source tweets and basic metadata.
    Expected layout (as in this repo):
      base_path/
        threads/<lang>/<event>/<thread_id>/source-tweets/<thread_id>.json
    Also attempts to read per-thread annotation.json for is_rumour, etc.
    """
    threads_root = os.path.join(base_path, 'threads')
    if not os.path.isdir(threads_root):
        raise FileNotFoundError(
            f"Couldn't find 'threads' directory at {threads_root}. Verify data_path.")
    rows = []
    for lang in os.listdir(threads_root):
        lang_path = os.path.join(threads_root, lang)
        if not os.path.isdir(lang_path):
            continue
        for event in os.listdir(lang_path):
            event_path = os.path.join(lang_path, event)
            if not os.path.isdir(event_path):
                continue
            for thread_id in os.listdir(event_path):
                thread_path = os.path.join(event_path, thread_id)
                if not os.path.isdir(thread_path):
                    continue
                src_dir = os.path.join(thread_path, 'source-tweets')
                src_file = os.path.join(src_dir, f'{thread_id}.json')
                if not os.path.isfile(src_file):
                    # Some datasets may slightly differ; skip if missing
                    continue
                try:
                    with open(src_file, 'r', encoding='utf-8') as f:
                        tweet = json.load(f)
                except Exception as e:
                    # Skip corrupted files but keep going
                    # You can log e if needed
                    continue
                # Try to enrich with per-thread annotation if available
                annot_file = os.path.join(thread_path, 'annotation.json')
                if os.path.isfile(annot_file):
                    try:
                        with open(annot_file, 'r', encoding='utf-8') as af:
                            annot = json.load(af)
                        # Add a few useful fields if present
                        tweet['is_rumour'] = annot.get('is_rumour')
                        tweet['veracity_true'] = annot.get('true')
                        tweet['misinformation'] = annot.get('misinformation')
                    except Exception:
                        pass
                tweet['language'] = lang
                tweet['event'] = event
                tweet['thread_id'] = thread_id
                rows.append(tweet)
    return pd.DataFrame(rows)
# Load dataset
df = load_pheme_data(data_path)
if df.empty:
    print('No source tweets were loaded. Check that the dataset path is correct and matches the expected structure.')
else:
    # Show general description
    print("Number of source tweets:", df.shape[0])
    print("Variables (columns):", list(df.columns))
    print("Data types:\n", df.dtypes)
    print("Missing values (per column):\n", df.isnull().sum())
    print("\nSample rows:\n", df.head())