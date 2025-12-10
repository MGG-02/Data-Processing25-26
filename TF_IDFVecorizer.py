import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def save_tfidf_to_csv(tfidf_matrix, feature_names, doc_ids, output_file="tfidf_matrix.csv"):

    # Convert sparse → dense safely
    tfidf_dense = tfidf_matrix.toarray()

    df = pd.DataFrame(tfidf_dense, index=doc_ids, columns=feature_names)

    df.to_csv(output_file)
    print(f"TF-IDF CSV saved: {output_file}")

    return df

# Carga los datos preprocesados desde el archivo CSV
data_path = 'csv_Dataset.csv'
pheme_df = pd.read_csv(data_path)


# Definir el vectorizador TF-IDF y ajustar a los resúmenes preprocesados
max_features = 1000
tfidf_vectorizer = TfidfVectorizer(max_features=max_features)  # Usamos las 1000 palabras más importantes
X_tfidf = tfidf_vectorizer.fit_transform(pheme_df['text'])
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convertir a DataFrame para una mejor visualización y manipulación
X_tfidf_df = save_tfidf_to_csv(X_tfidf, feature_names, pheme_df['thread_id'])

# Mostrar las características y algunos valores de TF-IDF
print(X_tfidf_df.head())

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
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score

X = X_tfidf_df.values
Y = pheme_df['target']

le = LabelEncoder()
Y = le.fit_transform(Y)     #type: ignore

X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y, test_size=0.30, random_state=42, stratify=Y
)

# Second split: Validation + Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print('\n'+'### ---  Logistic Regression --- ###' + '\n')

clf = LogisticRegression(solver="lbfgs",class_weight="balanced",max_iter=100)
clf.fit(X_train, y_train)
y_val_pred = clf.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

y_test_pred = clf.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_test_pred)}')
print(f'Test Roc Auc Score: {roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')}')
print(f'Test F1 Score: {f1_score(y_test, y_test_pred, average='micro')}')

print('\n' + '### --- SVM Classifier --- ###' + '\n')

Lclas = LinearSVC()
clf = CalibratedClassifierCV(Lclas).fit(X_train, y_train)
y_val_pred = clf.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

y_test_pred = clf.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_test_pred)}')
print(f'Test Roc Auc Score: {roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')}')
print(f'Test F1 Score: {f1_score(y_test, y_test_pred, average='micro')}')

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
print(f'Test F1 Score: {f1_score(y_test, y_test_pred, average='micro')}')

#####################################
# -----      Pytorch NN       ----- #
#####################################
from TF_IDF_torch_clf import *
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

batch_size = 1024
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=True)

epochs = 3000
model = tfidfClassifier(max_features)
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
plt.show()

evaluate(model, test_loader)