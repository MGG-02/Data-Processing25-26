import os
import json
import pandas as pd

from word_cleanner import clean_text
from convert_annotations_data import *

import os
import json
import pandas as pd


def load_dataset(base_path="./all-rnr-annotated-threads"):
    rows = []

    for event in os.listdir(base_path):
        event_path = os.path.join(base_path, event)
        if not os.path.isdir(event_path):
            continue

        for class_folder in ["rumours", "non-rumours"]:
            class_path = os.path.join(event_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            for thread_id in os.listdir(class_path):
                thread_path = os.path.join(class_path, thread_id)
                if not os.path.isdir(thread_path):
                    continue


                source_path = os.path.join(thread_path, "source-tweets", f"{thread_id}.json")
                annot_path = os.path.join(thread_path, "annotation", f"{thread_id}.json")

                if os.path.exists(source_path) and os.path.exists(annot_path):

                    with open(source_path) as f:
                        src = json.load(f)

                    with open(annot_path) as f:
                        ann = json.load(f)

                    label = convert_annotations_data(ann)

                    rows.append({
                        "tweet_id": thread_id,
                        "thread_id": thread_id,
                        "event": event,
                        "is_source": True,
                        "text": src.get("text", ""),
                        "cleaned_text": clean_text(src.get("text", "")),
                        "label_veracity": label,
                        "parent_tweet_id": None
                    })


                replies_dir = os.path.join(thread_path, "reactions")
                if os.path.isdir(replies_dir):
                    for filename in os.listdir(replies_dir):

                        if not filename.endswith(".json"):
                            continue

                        reply_id = filename.replace(".json", "")
                        reply_path = os.path.join(replies_dir, filename)
                        reply_annot = os.path.join(thread_path, "annotation", f"{reply_id}.json")

                        with open(reply_path, "r", encoding="utf-8", errors="replace") as f:
                            reply = json.load(f)


                        if not os.path.exists(reply_annot):
                            continue  # skip replies without annotations

                        with open(reply_annot) as f:
                            ann = json.load(f)

                        label = convert_annotations_data(ann)

                        rows.append({
                            "tweet_id": reply_id,
                            "thread_id": thread_id,
                            "event": event,
                            "is_source": False,
                            "text": reply.get("text", ""),
                            "cleaned_text": clean_text(reply.get("text", "")),
                            "label_veracity": label,
                            "parent_tweet_id": reply.get("in_reply_to_status_id_str")
                        })

    df = pd.DataFrame(rows)
    print(f"Loaded total tweets (source + replies): {len(df)}")
    return df