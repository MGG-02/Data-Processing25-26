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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Label array ('True', 'False', 'Unverified')
Y = pheme_df['target'].values

# 60% -> Training dataset, 40% -> Test/validation datasets
X_train,X_temp,y_train,y_temp = train_test_split(X_w2v,Y,test_size=0.4,random_state=42,stratify=Y) #type: ignore

# Split test/validation dataset 50/50 (20% -> Test dataset, 20% -> Validation dataset)
X_val,X_test,y_val,y_test = train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)

print("Datasets size")
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

