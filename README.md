# Analysis of Ideological Polarization on Social Networks Based on the Spread of Disinformation Content
### Final Project – Data Processing (Master in Telecommunication Engineering)  
### December 2025

### Manuel Garde Granizo, Jose Ignacio Aguilar Anguita

---

## 1. Problem Description

The spread of disinformation on social networks has a strong impact on public perception and can reinforce ideological polarization. Misleading or unverified information tends to propagate quicker than verified content, especially within ideologically homogeneous communities.

**Objective:**  
To analyze how rumour-like (disinformative) content spreads and whether it reflects linguistic patterns associated with polarization. This includes:

- Detecting whether posts are **true**, **false**, or **unverified**  
- Comparing several text-vectorization strategies  
- Evaluating classical ML models, neural networks, and fine-tuned transformers  
- Interpreting whether rumour content shows linguistic polarization cues (emotion, stance, uncertainty)

---

## 2. Dataset Description

We use the **PHEME Rumour Scheme Dataset**, containing Twitter posts annotated as **true**, **false**, or **unverified**.  
It includes source tweets and full reply threads across multiple major news events.

### Dataset Summary  
- **Total posts:** *[fill in]*  
- **Events:** e.g., Charlie Hebdo, Sydney Siege, Germanwings Crash  
- **Classes:**  
  - True: *[n]*  
  - False: *[n]*  
  - Unverified: *[n]*  
- **Average post length:** *[n]* tokens  

### Exploratory Analysis  
- Distribution of classes  
- Tweet length distribution  
- Word frequency analysis and word clouds  
- Example tweets per class  

### Initial Hypotheses  
1. False and unverified posts use more uncertain or emotional vocabulary.  
2. Transformer-based embeddings will outperform TF-IDF and Word2Vec.  
3. The *unverified* class will be the most difficult to classify.

---

## 3. Methodology

### 3.1 Preprocessing  
- Text cleaning (URLs, mentions, punctuation)  
- Lowercasing and emoji normalization  
- Tokenization via HuggingFace tokenizer  
- Stratified train/validation/test split (70/15/15)  

---

## 3.2 Text Vector Representations

We compared three embedding strategies:

### **A) TF-IDF**
- Vocabulary size: 1,000–5,000  
- Unigrams + bigrams  
- Produces sparse document vectors  

### **B) Word2Vec**
- Pretrained embeddings or self-trained skip-gram model  
- Document vector = average of word embeddings  

### **C) Transformer-Based Embeddings**
(Using BERT, RoBERTa, or BERTweet)
- CLS token representation  
- Mean pooling of final hidden layer  
- Provides contextual, dynamic embeddings  

---

## 3.3 Classification Models

### **Classical Models (Scikit-Learn)**
Trained on TF-IDF and Word2Vec vectors:
- Logistic Regression  
- SVM  
- Random Forest  
- (Optional) KNN  

### **PyTorch Neural Network**
Architecture:

Input → Dense(256) → ReLU → Dropout → Dense(3) → Softmax

### References
<a id="1">[1]</a> 
Kochkina, Elena; Liakata, Maria; Zubiaga, Arkaitz (2018). 
PHEME dataset for Rumour Detection and Veracity Classification. figshare.
[Dataset](https://doi.org/10.6084/m9.figshare.6392078.v1)
