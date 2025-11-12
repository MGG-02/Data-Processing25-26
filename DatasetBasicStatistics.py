import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DatasetGeneralDescription import load_pheme_data

path = "pheme-rumour-scheme-dataset"
df = load_pheme_data(path)

# compute text length
df['text_length'] = df['text'].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)

# Rumour bar plot (you already have this)
ax = df['is_rumour'].value_counts(dropna=False).sort_index().plot(
    kind='bar', color='#4c78a8', edgecolor='black')
ax.set_xlabel('is_rumour')
ax.set_ylabel('count')
ax.set_title('Counts of is_rumour')
plt.tight_layout()
plt.show()

# Histogram (matplotlib only)
data = df['text_length'].dropna().values
plt.figure(figsize=(8,4))
plt.hist(data, bins=30, edgecolor='black', alpha=0.75)   # counts on y-axis
plt.title("Source Tweet Text Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
