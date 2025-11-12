import os
import json
import pandas as pd
from DatasetGeneralDescription import load_pheme_data
import matplotlib.pyplot as plt
import seaborn as sns

# Set this to your "threads" directory
path = "pheme-rumour-scheme-dataset"

df = load_pheme_data(path)
print("\nRumour/Non-rumour distribution:")
print(df['is_rumour'].value_counts(dropna=False))

# Threads per language and event
print("\nThreads per language:")
print(df['language'].value_counts())

print("\nThreads per event:")
print(df['event'].value_counts())

# Text length stats (words)
df['text_length'] = df['text'].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)

plt.figure(figsize=(6,4))
sns.countplot(x='is_rumour', data=df)
plt.title("Rumour vs Non-Rumour Thread Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Threads")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title("Source Tweet Text Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()
