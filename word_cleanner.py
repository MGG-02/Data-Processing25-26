import re
import lxml
import contractions
from bs4 import BeautifulSoup
import re

def clean_words(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-zA-Z]", " ", text)
    return text.split()

def clean_text(text: str) -> str:
    
    import re
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

import nltk

def check_nltk_packages():
  packages = ['punkt','stopwords','omw-1.4','wordnet']

  for package in packages:
    try:
      nltk.data.find('tokenizers/' + package)
    except LookupError:
      nltk.download(package)
check_nltk_packages()

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.tokenize import wordpunct_tokenize

def prepare_data(text):

    text = contractions.fix(text)                                 # expand contractions
    soup = BeautifulSoup(text, "lxml")                            # remove HTML
    text = soup.get_text(separator=" ")
    text = re.sub(r'https?://\S+|www\.\S+', '', text)             # remove URLs

    tokens = wordpunct_tokenize(text)

    # flatten list of lists
    tokens = [tok for sent in tokens for tok in sent]

    # lowercase + keep only alphanumeric tokens
    tokens = [tok.lower() for tok in tokens if tok.isalnum()]

    stemmer = SnowballStemmer('english')
    wnl = WordNetLemmatizer()
    # lemmatize
    tokens = [wnl.lemmatize(tok) for tok in tokens]

    stopwords_en = stopwords.words('english')
    # remove stopwords
    tokens = [tok for tok in tokens if tok not in stopwords_en]
