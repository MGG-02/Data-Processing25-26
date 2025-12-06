import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformer_model import test_df

checkpoint_path = "./finetuned-model/checkpoint-6724/"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()

test_dataset = Dataset.from_pandas(test_df)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.remove_columns(["text"])
test_dataset.set_format(type="torch")

from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=32)        #type:ignore
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(batch["labels"].numpy())

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("\nF1 (macro):", f1_score(all_labels, all_preds, average="macro"))
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["false", "true", "unverified"]))