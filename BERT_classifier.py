import random
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
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

from sklearn.model_selection import train_test_split

X = X.cpu().numpy()

#Create Train, Validation and Test splits

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y) #type: ignore

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)

print(f'Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

reg = LogisticRegression(max_iter=10000)
reg.fit(X_train, y_train)

y_val_pred = reg.predict(X_val)
print("Validation report (LogReg on BERT embeddings):")
print(classification_report(y_val, y_val_pred))

# Final eval on test set
y_test_pred = reg.predict(X_test)
print("Test report (LogReg on BERT embeddings):")
print(classification_report(y_test, y_test_pred))

