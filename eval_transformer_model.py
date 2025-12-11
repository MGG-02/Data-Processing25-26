import torch
from transformer_data_prepro import preprocess_data
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint_path = "./finetuned-model/checkpoint-6724/"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()

dataset, tokenizer, _ = preprocess_data()

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

print(dataset)

from torch.utils.data import DataLoader

test_loader = DataLoader(dataset['test'], batch_size=32)        #type:ignore
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
        all_labels.extend(batch['labels'].numpy())

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("\nF1 (macro):", f1_score(all_labels, all_preds, average="macro"))
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["false", "true", "unverified"]))