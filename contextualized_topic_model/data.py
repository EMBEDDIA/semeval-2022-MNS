import string
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

exclude = set(string.punctuation)
stops = set(stopwords.words('english'))


def clean_document(doc, lang='en'):
    clean_punc = ''.join(ch if ch not in exclude else ' ' for ch in doc.lower())
    clean_punc_tokens = clean_punc.split()
    clean_stop = [tok for tok in clean_punc_tokens if tok not in stops]
    clean_digits = [tok for tok in clean_stop if re.search(r'\d', tok) is None]
    clean_short = [tok for tok in clean_digits if 2 < len(tok) < 20]
    return clean_short


def prune_vocabulary(documents, min_df=5, max_df=0.8, min_len=20):
    print("Truncating vocab with min_word_freq =", min_df, "and max_doc_prop =", max_df)
    docs = [" ".join(doc) for doc in documents]
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stops)
    cvectorizer.fit_transform(docs).sign()
    dictionary = list(cvectorizer.vocabulary_)
    print("Truncated vocab size:", len(dictionary))
    pruned_documents = []
    for doc in documents:
        pruned_doc = [w for w in doc if w in dictionary]
        #if len(pruned_doc) >= min_len:
        pruned_documents.append(pruned_doc)
    return pruned_documents


def prune_vocabulary2(documents_raw, min_df=5, max_df=0.8, min_len=20):
    print("Truncating vocab with min_word_freq =", min_df, "and max_doc_prop =", max_df)
    #docs = [" ".join(doc) for doc in documents]
    documents_unproc = []
    documents_proc = [' '.join(clean_document(doc.lower())) for doc in documents_raw]
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stops)
    cvectorizer.fit_transform(documents_raw).sign()
    dictionary = list(cvectorizer.vocabulary_)
    print("Truncated vocab size:", len(dictionary))
    for i in range(len(documents_proc)):
        pruned_doc = [w for w in documents_proc[i] if w in dictionary]
        if len(pruned_doc) >= min_len:
            documents_proc[i] = pruned_doc
            documents_unproc[i] = documents_raw[i].lower()
    return documents_unproc, documents_proc