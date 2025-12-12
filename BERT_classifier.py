import random
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from tabulate import tabulate
from load_dataset import *

random_seed = 42
random.seed(random_seed)

#just keep text and targets for BERT
labels_to_drop = ['event',
                  'thread_id',
                  'tweet_id',
                  'date',
                  'fav_counts',
                  'retweet_counts',
                  'username',
                  'account_date',
                  'protected',
                  'verified',
                  'followers',
                  'friends_count',
                  'followings',
                  'hashtag',
                  'urls']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device) # type: ignore
model.eval()

#Prepare data to create embbedings
data_path = 'csv_Dataset.csv'
pheme_df = pd.read_csv(data_path).drop(labels_to_drop, axis=1)
texts = pheme_df['text'].astype(str).to_list()

texts = [clean_text(i) for i in texts]

# The function returns a dictionary containing the token IDs and attention masks
encoding = tokenizer(
    texts,                   # List of input texts
    padding=True,              # Pad to the maximum sequence length
    truncation=True,           # Truncate to the maximum sequence length if necessary
    return_tensors='pt',      # Return PyTorch tensors
    add_special_tokens=True    # Add special tokens CLS and SEP
)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)
print(f'Input IDs: {input_ids[0]}', f'Attention mask: {attention_mask[0]}')

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    word_embeddings = outputs.last_hidden_state  # This contains the embeddings

print(f'Shape of Word Embeddings: {word_embeddings.shape}')



#Mean Poolng for classification tasks
mask_expanded = attention_mask.unsqueeze(-1)                    # [2402, 44, 1]
masked_embeddings = word_embeddings * mask_expanded            # zero out padding tokens

sum_embeddings = masked_embeddings.sum(dim=1)                   # [2402, 768]
sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)             # [2402, 1]

sentence_embeddings = sum_embeddings / sum_mask                 # [2402, 768]

X = sentence_embeddings
Y = pheme_df['target'].values


                                                    #####################################
                                                    # ----- CLASSIFICATION MODELS ----- #
                                                    #####################################


#####################################
# -----   Scikit-learn ALG    ----- #
#####################################

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

le = LabelEncoder()
Y = le.fit_transform(Y)     #type: ignore


X = X.cpu().numpy()

#Create Train, Validation and Test splits

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y) #type: ignore

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)

print('\n'+'### ---  Logistic Regression --- ###' + '\n')

clf = LogisticRegression(solver="lbfgs",class_weight="balanced",max_iter=300)
clf.fit(X_train, y_train)
y_val_pred = clf.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

y_test_pred = clf.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_test_pred)}')
print(f'Test Roc Auc Score: {roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')}')
print(f'Test F1 Score (Weighted): {f1_score(y_test, y_test_pred, average="weighted"):.4f}')
print(f'Test F1 Score (Macro): {f1_score(y_test, y_test_pred, average="macro"):.4f}')
print("-" * 30)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

print('\n' + '### --- SVM Classifier --- ###' + '\n')

Lclas = LinearSVC()
clf = CalibratedClassifierCV(Lclas).fit(X_train, y_train)
y_val_pred = clf.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

y_test_pred = clf.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_test_pred)}')
print(f'Test Roc Auc Score: {roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')}')
print(f'Test F1 Score (Weighted): {f1_score(y_test, y_test_pred, average="weighted"):.4f}')
print(f'Test F1 Score (Macro): {f1_score(y_test, y_test_pred, average="macro"):.4f}')
print("-" * 30)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

print('\n'+'### ---  Random Forest Classification --- ###' + '\n')

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
).fit(X_train, y_train)

y_val_pred = rf.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

y_test_pred = rf.predict(X_test)
y_test_proba = rf.predict_proba(X_test)

print(f'Test Accuracy: {accuracy_score(y_test, y_test_pred)}')
print(f"ROC AUC Score for test SET: {roc_auc_score(y_test, y_test_proba, multi_class='ovr')}")
print(f'Test F1 Score (Weighted): {f1_score(y_test, y_test_pred, average="weighted"):.4f}')
print(f'Test F1 Score (Macro): {f1_score(y_test, y_test_pred, average="macro"):.4f}')
print("-" * 30)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))



#####################################
# -----      Pytorch NN       ----- #
#####################################
from BERT_torch_clf import *
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)

#Normalize inputs for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=True)

epochs = 500
model = BertClassifier()
LR = 5e-6
print('\n'+'### ---  PYTORCH NN --- ###' + '\n')
train(model, train_loader, val_loader, LR, epochs)
epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(12, 5))

# ---- LOSS CURVE ----
plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

# ---- ACCURACY CURVE ----
plt.subplot(1, 2, 2)
plt.plot(epochs, history["train_acc"], label="Train Acc")
plt.plot(epochs, history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig('BERT_NN_trainig.png')
plt.show()

evaluate(model, test_loader)
