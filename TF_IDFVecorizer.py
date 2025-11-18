
def save_tfidf_to_csv(tfidf_matrix, feature_names, doc_ids, output_file="tfidf_matrix.csv"):

    # Convert sparse → dense safely
    tfidf_dense = tfidf_matrix.toarray()

    df = pd.DataFrame(tfidf_dense, index=doc_ids, columns=feature_names)

    df.to_csv(output_file)
    print(f"TF-IDF CSV saved: {output_file}")
    print(f"Shape: {df.shape}")

    return df

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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