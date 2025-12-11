import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

label_map = {"false": 0, "true": 1, "unverified": 2}

def preprocess_data(csv_path="csv_Dataset.csv", model_name="vinai/bertweet-base"):
    # Load raw CSV
    df = pd.read_csv(csv_path)

    # Split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["target"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["target"]
    )

    # Build HuggingFace Dataset
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "val": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Keep only text + label
    dataset = dataset.remove_columns(
        [col for col in dataset["train"].column_names if col not in ["text", "target"]]
    )
    dataset = dataset.rename_columns({"target": "labels"})

    # Encode labels
    def encode_labels(batch):
        batch["labels"] = label_map[batch["labels"]]
        return batch

    dataset = dataset.map(encode_labels)

    # Tokenization
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)

    return dataset, tokenizer, (train_df, val_df, test_df)
