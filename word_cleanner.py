import re

def clean_words(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-zA-Z]", " ", text)
    return text.split()
