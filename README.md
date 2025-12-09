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

We use the **PHEME Rumour Scheme Dataset** [[1]](#1), containing Twitter posts annotated as **Missinformation** (True/False) and **True** (0/1).
In order to provide veracity of the Tweet, those labels are mapped into:

- Missinformation & True == 0 → "Unverified"
- Missinformation == 0 & True == 1 → "Verified"
- Missinformation == 1 & True == 0 → "False"
- Missinformation == 1 & True == 1 → "ERROR!!"

It includes source tweets and full reply threads across multiple major news events.

### Dataset Summary  
- **Total posts:** 2402 Rumour Tweets  
- **Events:** e.g., Charlie Hebdo, Sydney Siege, Germanwings Crash  
- **Classes:**  
  - True: *1067*  
  - False: *638*  
  - Unverified: *697*  
- **Average post length:** *[n]* tokens  

### Initial Hypotheses  
1. False and unverified posts use more uncertain or emotional vocabulary.  
2. Transformer-based embeddings will outperform TF-IDF and Word2Vec.  
3. The *unverified* class will be the most difficult to classify.

---

## 3. Methodology

Copy repo into local machine:

<pre> bash: ~$ git clone https://github.com/MGG-02/Data-Processing25-26.git </pre>

### 3.1 Exploratory Analysis  

<pre> bash: ~$ python3 DataBasicStats-GenDescript.py </pre>

- Dataset dimentions, data types and dataset samples  
- Missing values  
- Basic stats for numerical values (count, mean, std)  
- Target Distribuition and samples for each veracity label
- WordCloud  

---

## 3.2 Text Vector Representations

Three different text vectorization strategies are compared:

### **A) TF-IDF**

_Term Frequency-Inverse Document Frequency_ is a text vectorization technique that expresses how relevant a word in a document is. Given by its formula:

$$TF = \frac{BoW (w, d)}{\text{nº words in d}} \space \space ; \space \space IDF = \log{\frac{\text{nº Docs}}{\text{nº Docs with term w}}}$$

$$TF-IDF = TF(w,d) \times IDF (w)$$

The main purpouse for TF-IDF vectorization is counting words and deciding weather a word is more or less important in order to detect the topic/sentimient/etc.

### **B) Word2Vec**
- Pretrained embeddings or self-trained skip-gram model  
- Document vector = average of word embeddings  

### **C) Transformer-Based Embeddings**
**BERT** Embeddings: The previous text vectorization techniques are unable to capture context in the sentences, this is why, the last vectorization used is BERT. BERT (_Bidirectional Encoder Representations from Transformers_) is a pretrained language model that uses bidirectional context to enhance performance on natural language processing tasks.[[2]](#2)

![Project Logo](images/bert-embedding-layer.png)

Because BERT captures nuanced semantic and emotional signals, it is particularly well-suited for detecting disinformation patterns and ideological polarization in social media posts.

BERT excels at:
  - Capturing subtle sentiment and linguistic cues
  - Understanding stance, irony, or emotionally charged language
  - Differentiating between factual vs. manipulative phrasing
  - Modeling ideologically polarized vocabulary shifts
  
---

## 3.3 Classification Models

### **Classical Models (Scikit-Learn)**

In order to create models able to classify weather a Tweet is _True_, _False_ or _Unverified_, several Scikit-Learn Classifiers have been selected:

- Logistic Regression
- SVM  
- Random Forest  

| Model | Type | Captures Nonlinearity | Pros | Cons | Works Well With |
|-------|------|------------------------|------|------|------------------|
| **Logistic Regression** | Linear | No | Simple, fast, interpretable, strong baseline | Limited for complex patterns | High-dimensional embeddings (TF-IDF, BERT) |
| **SVM** | **Linear**/Nonlinear | Yes (with kernels) | Strong performance, robust to outliers | Slow on large datasets, requires tuning | High-dimensional features (especially linear kernel) |
| **Random Forest** | Tree Ensemble | Yes | Captures complex patterns, robust to noise, feature importance | Weak on dense high-dimensional embeddings, can overfit | Tabular data or sparse text vectors |

### **PyTorch Neural Network**

To explore different ways of Tweet classification/Spread of desinformation, a Pytorch Neural Network has been created. Neural Networks (**NN**) can learn nonlinear relationships that scikit-learn models struggle with. This is usefull when working with text representation via embeddings, where context vectors dimentions are complex.

Another advantage of NN is the high customization of the model, where dimension and number of layers, activation functions, dropout, loss function, optimizers, etc can be tunned in order to increase performance, reduce loss and increase accuracy.

To end up with the advantages of NN, Pytorch provides GPU acceleration, meaning that the created model can be tranfered to the GPU to increase training, validation and test speed.

Architecture:

![Project Logo](images/NN-1.png)

### References
<a id="1">[1]</a> 
Kochkina, Elena; Liakata, Maria; Zubiaga, Arkaitz (2018). 
PHEME dataset for Rumour Detection and Veracity Classification. figshare.
[Dataset](https://doi.org/10.6084/m9.figshare.6392078.v1)

<a id="2">[2]</a> 
David Liang (2024)
Intro — Getting Started with Text Embeddings: Using BERT
[](https://medium.com/@davidlfliang/intro-getting-started-with-text-embeddings-using-bert-9f8c3b98dee6)
