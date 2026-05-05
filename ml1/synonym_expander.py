import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

def expand_with_synonyms(text):
    words = text.split()
    expanded = set(words)

    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))

    return " ".join(expanded)
