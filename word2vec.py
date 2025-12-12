# --- Word2vec text vectorization --- #

import numpy as np
import pandas as pd
from word_cleanner import clean_words
from gensim.models import Word2Vec

# Load dataset 
data_path = 'csv_Dataset.csv'
pheme_df = pd.read_csv(data_path)

# Tokenize each word in each tweet
texts = pheme_df['text'].astype(str).to_list()
tokenized_tweets = [clean_words(i) for i in texts]

# Train model using tokens
model_w2v = Word2Vec(tokenized_tweets,min_count=1,vector_size=100,window=5,workers=4,sg=1)

# Mean of each tweet's words embeddings
def tweet_vector(tokens,model):
   return np.mean(model.wv[tokens],axis=0)

X_w2v = np.array([tweet_vector(tweet_tokens,model_w2v) for tweet_tokens in tokenized_tweets])

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
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Label array ('True', 'False', 'Unverified')
Y = pheme_df['target'].values

# Label Encoding ('True' -> 0, 'False' -> 1, 'Unverified' -> 2)
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y) #type:ignore

# 60% -> Training dataset, 40% -> Test/validation datasets
X_train,X_temp,y_train,y_temp = train_test_split(X_w2v,Y_encoded,
      test_size=0.4,random_state=42,stratify=Y_encoded) #type: ignore

# Split test/validation dataset 50/50 (20% -> Test dataset, 20% -> Validation dataset)
X_val,X_test,y_val,y_test = train_test_split(X_temp,y_temp,
      test_size=0.5,random_state=42,stratify=y_temp)

# Standarize datasets
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

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
from W2V_torch_clf import *
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
model = Word2VecClassifier(input_dim=X_w2v.shape[1], 
                           num_classes=len(label_encoder.classes_))
LR = 5e-4
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
plt.savefig('W2V_NN_trainig.png')
plt.show()

evaluate(model, test_loader)