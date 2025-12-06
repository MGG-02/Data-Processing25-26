#We will use roberta-base because is trained along more data and longer sequences

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,  #Loading this instead of RoBERTa directly is usefull to avoid implementing your own classifier, as it already provides it
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch 
from sklearn.metrics import accuracy_score

#AutoModelFrSeq... classifier is already composed by a Dense Layer, Dropout Layer and a Softmax Layer

data_path = 'csv_Dataset.csv'
df = pd.read_csv(data_path)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["target"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["target"])

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "val": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
    }) # type: ignore

#https://huggingface.co/transformers/v3.0.2/model_doc/roberta.html#transformers.RobertaConfig
model1_name = "roberta-base"

#https://github.com/VinAIResearch/BERTweet
model2_name = "vinai/bertweet-base"

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["text", "target"]])
dataset = dataset.rename_columns(column_mapping = {'target': 'labels'})


def tokenize(batch):
    return tokenizer(batch['text'],
                     truncation = True,
                     padding = "max_length",
                     max_length = 128)

label_map = {"false": 0, "true": 1, "unverified": 2}

def encode_labels(example):
    example["labels"] = label_map[example["labels"]]
    return example

dataset = dataset.map(encode_labels)
dataset = dataset.map(tokenize, batched=True)

print(dataset)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(model2_name,num_labels=3)

training_args = TrainingArguments(
    output_dir="finetuned-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    seed=42,
    bf16=True
)

def compute_metrics(eval_preds):
    '''Evaluation function based on accuracy'''
    pred, labels = eval_preds
    pred = np.argmax(pred, axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['val'],
    processing_class = tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)]
)

trainer.train()

results = trainer.evaluate()   # just gets evaluation metrics
print(results)