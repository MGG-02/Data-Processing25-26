#We will use roberta-base because is trained along more data and longer sequences

import pandas as pd
import numpy as np
from transformer_data_prepro import preprocess_data
from transformers import (
    AutoModelForSequenceClassification,  #Loading this instead of RoBERTa directly is usefull to avoid implementing your own classifier, as it already provides it
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
) 
from sklearn.metrics import accuracy_score

#AutoModelFrSeq... classifier is already composed by a Dense Layer, Dropout Layer and a Softmax Layer

#https://huggingface.co/transformers/v3.0.2/model_doc/roberta.html#transformers.RobertaConfig
model1_name = "roberta-base"

#https://github.com/VinAIResearch/BERTweet
model2_name = "vinai/bertweet-base"

dataset, tokenizer, _ = preprocess_data()

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
    seed=42
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

# Plotting
import matplotlib.pyplot as plt

log_history = trainer.state.log_history

train_loss = []
train_epochs = []
val_loss = []
val_acc = []
val_epochs = []

for entry in log_history:
    if 'loss' in entry and 'epoch' in entry:
        train_loss.append(entry['loss'])
        train_epochs.append(entry['epoch'])
    if 'eval_loss' in entry and 'epoch' in entry:
        val_loss.append(entry['eval_loss'])
        val_acc.append(entry['eval_accuracy'])
        val_epochs.append(entry['epoch'])

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_epochs, train_loss, label='Train Loss')
plt.plot(val_epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(val_epochs, val_acc, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('transformer_training.png')
print("Figure saved as transformer_training.png")

results = trainer.evaluate()   # just gets evaluation metrics
print(results)