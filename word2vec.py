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

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

######## Logistic Regression ########

from sklearn.linear_model import LogisticRegression

# Label array ('True', 'False', 'Unverified')
Y = pheme_df['target'].values

# Label Encoding ('True' -> 0, 'False' -> 1, 'Unverified' -> 2)
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

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

print("Scikit - Datasets size")
print("Train:", X_train.shape, "\nVal:", X_val.shape, "\nTest:", X_test.shape)

# Multiclass Logistic Regression model
log_reg = LogisticRegression(solver="lbfgs",class_weight="balanced",max_iter=100)

# Model training with training dataset (X_train,y_train)
log_reg.fit(X_train,y_train)

# Validation report over validation dataset (X_val,y_val)
y_val_pred = log_reg.predict(X_val)
print("Validation report (LogReg on W2V embeddings):")
print(classification_report(y_val, y_val_pred))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# Final evaluation over test dataset (X_test,y_test)
y_test_pred = log_reg.predict(X_test)
print("Test report (LogReg on W2V embeddings):")
print(classification_report(y_test, y_test_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Roc Auc Score: ", roc_auc_score(y_test, log_reg.predict_proba(X_test), multi_class='ovr'))
print("Test F1 Score: ", f1_score(y_test, y_test_pred, average='micro'))

################ SVM ################

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# SVM model
Lclas = LinearSVC()

# Model training with training dataset (X_train,y_train)
clf = CalibratedClassifierCV(Lclas).fit(X_train, y_train)

# Validation report over validation dataset (X_val,y_val)
y_val_pred_svm = clf.predict(X_val)
print("Validation report (SVM on W2V embeddings):")
print(classification_report(y_val, y_val_pred_svm))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred_svm))

# Final evaluation over test dataset (X_test,y_test)
y_test_pred_svm = clf.predict(X_test)
print("Test report (SVM on W2V embeddings):")
print(classification_report(y_test, y_test_pred_svm))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred_svm))
print("Test Roc Auc Score: ", roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr'))
print("Test F1 Score: ", f1_score(y_test, y_test_pred_svm, average='micro'))

########### Random Forest ###########

from sklearn.ensemble import RandomForestClassifier

# Random Forest model
rf_clf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2,
      min_samples_leaf=1, class_weight='balanced_subsample', n_jobs=-1, random_state=42)

# Model training with training dataset (X_train,y_train)
rf_clf.fit(X_train, y_train)

# Validation report over validation dataset (X_val,y_val)
y_val_pred_rf = rf_clf.predict(X_val)
print("Validation report (Random Forest on W2V embeddings):")
print(classification_report(y_val, y_val_pred_rf))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred_rf))

# Final evaluation over test dataset (X_test,y_test)
y_test_pred_rf = rf_clf.predict(X_test)
print("Test report (Random Forest on W2V embeddings):")
print(classification_report(y_test, y_test_pred_rf))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print("Test Roc Auc Score: ", roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr'))
print("Test F1 Score: ", f1_score(y_test, y_test_pred_rf, average='micro'))

##########################################
# -----   PyTorch NN CLASSIFIER    ----- #
##########################################

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from W2V_torch_clf import *

# 60% -> Training dataset, 40% -> Test/validation datasets
X_train_nn,X_temp_nn,y_train_nn,y_temp_nn = train_test_split(X_w2v,Y_encoded,test_size=0.4,random_state=42,stratify=Y_encoded) #type: ignore

# Split test/validation dataset 50/50 (20% -> Test dataset, 20% -> Validation dataset)
X_val_nn,X_test_nn,y_val_nn,y_test_nn = train_test_split(X_temp_nn,y_temp_nn,test_size=0.5,random_state=42,stratify=y_temp_nn)

print("PyTorch NN - Datasets size")
print("Train:", X_train_nn.shape, "\nVal:", X_val_nn.shape, "\nTest:", X_test_nn.shape)

# Standarize datasets
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train_nn)
X_val_nn = scaler.transform(X_val_nn)
X_test_nn = scaler.transform(X_test_nn)

# CPU / GPU processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# PyTorch tensors
X_train_t = torch.tensor(X_train_nn,dtype=torch.float32)
y_train_t = torch.tensor(y_train_nn,dtype=torch.long)

X_val_t = torch.tensor(X_val_nn,dtype=torch.float32)
y_val_t = torch.tensor(y_val_nn,dtype=torch.long)

X_test_t = torch.tensor(X_test_nn,dtype=torch.float32)
y_test_t = torch.tensor(y_test_nn,dtype=torch.long)

# Data Loaders
batch_size = 64

train_loader = DataLoader(TensorDataset(X_train_t,y_train_t),
                          batch_size=batch_size,shuffle=True)

val_loader = DataLoader(TensorDataset(X_val_t,y_val_t),
                        batch_size=batch_size,shuffle=False)

test_loader = DataLoader(TensorDataset(X_test_t,y_test_t),
                         batch_size=batch_size,shuffle=False)

# Model training
EPOCHS = 500
LR = 1e-3
w2v_model = Word2VecClassifier(input_dim=X_w2v.shape[1], 
                               num_classes=len(label_encoder.classes_))
train(w2v_model, train_loader, val_loader, learning_rate=LR, epochs=EPOCHS)

# Model evaluation
evaluate(w2v_model, test_loader)
