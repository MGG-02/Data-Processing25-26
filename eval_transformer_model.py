import torch
import numpy as np
from transformer_data_prepro import preprocess_data
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint_path = "./finetuned-model/checkpoint-6724/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)
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
all_probs = []

print(device)
batch = next(iter(test_loader))
print(type(batch))
print(batch)

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits                  # (batch, num_classes)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_probs  = np.concatenate(all_probs)


print("\nAccuracy:", accuracy_score(all_labels, all_preds))
print(f'Test Roc Auc Score: {roc_auc_score(all_labels, all_probs, multi_class='ovr')}')
print(f'Test F1 Score (Weighted): {f1_score(all_labels, all_preds, average="weighted"):.4f}')
print(f'Test F1 Score (Macro): {f1_score(all_labels, all_preds, average="macro"):.4f}')
print("-" * 30)
print("Classification Report:")
print(classification_report(all_labels, all_preds))