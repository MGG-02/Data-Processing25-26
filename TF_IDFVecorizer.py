import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def save_tfidf_to_csv(tfidf_matrix, feature_names, doc_ids, output_file="tfidf_matrix.csv"):

    # Convert sparse → dense safely
    tfidf_dense = tfidf_matrix.toarray()

    df = pd.DataFrame(tfidf_dense, index=doc_ids, columns=feature_names)

    df.to_csv(output_file)
    print(f"TF-IDF CSV saved: {output_file}")
    print(f"Shape: {df.shape}")

    return df

# Carga los datos preprocesados desde el archivo CSV
data_path = 'csv_Dataset.csv'
pheme_df = pd.read_csv(data_path)


# Definir el vectorizador TF-IDF y ajustar a los resúmenes preprocesados
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Usamos las 1000 palabras más importantes
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

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X = X_tfidf_df.values
Y = pheme_df['target']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y, test_size=0.30, random_state=42, stratify=Y
)

# Second split: Validation + Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)


Lclas = LinearSVC()
Lclas.fit(X_train, y_train)
y_val_pred = Lclas.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred))
y_test_pred = Lclas.predict(X_test)
print(y_test_pred)
print(accuracy_score(y_test, y_test_pred))

#####################################
# -----      Pytorch NN       ----- #
#####################################