'''
LexRank

see also:
"LexRank method for Text Summarization": https://iq.opengenus.org/lexrank-text-summarization/

----- 6 most relevant sentences (30%) -----
1 - Henry VIII was King of England from 1509 until his death in 1547.
2 - Henry's contemporaries considered him an attractive, educated, and accomplished king.
3 - His disagreement with Pope Clement VII on the question of such an annulment led Henry to initiate the English Reformation, 
    separating the Church of England from papal authority.
4 - Domestically, Henry is known for his radical changes to the English Constitution, ushering in the theory of the 
    divine right of kings.
5 - Henry was an extravagant spender, using the proceeds from the dissolution of the monasteries and acts of the 
    Reformation Parliament.
6 - At home, he oversaw the legal union of England and Wales with the Laws in Wales Acts 1535 and 1542, 
    and he was the first English monarch to rule as King of Ireland following the Crown of Ireland Act 1542.

'''

import json
import re
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


TOKENIZER = RegexpTokenizer(r"\w+")
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

VECTORIZER = TfidfVectorizer()


def _tokenize_text(text:str) -> List[str]:
    def filter_by_pos(token: List[str]) -> List[str]:  # filter by pos tags
        return [t for t, pos in pos_tag(token) if pos in ["NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]
    def remove_stopwords(token: List[str]) -> List[str]:
        return [w for w in token if not w in STOPWORDS]
    def tokenize_text(text: str) -> List[str]:
        return TOKENIZER.tokenize(text)  # list of token without punctuation etc.
    def get_token_lemmata(token: List[str]) -> List[str]:  # prefer over stemming
        return [LEMMATIZER.lemmatize(t) for t in token]
    def get_token_stems(token: List[str]) -> List[str]:
        return [STEMMER.stem(t) for t in token]
    
    token: List[str] = get_token_lemmata(remove_stopwords(tokenize_text(text.lower())))
    token = filter_by_pos(token)
    
    return token


with open("data/source_texts/wikipedia_a_1.txt") as f:
    loaded_text: str = f.read()

loaded_text = loaded_text.replace("\n\n", " ")

# ****************************************************
#
# text pre-processing
#
# ****************************************************
# wikipedia specific: remove paranthesis and brackets incl. text inside i.e. [1] or (28 June 1491 – 28 January 1547)
for x in set(re.findall("[\(\[].*?[\)\]]", loaded_text)):
    loaded_text = loaded_text.replace(x, "")
loaded_text = " ".join(loaded_text.split()).strip(".")  # remove double space


# ***
# split into sentences
# use for clean-up and later to match best results with original sentences
sentences: List[str] = loaded_text.split(".")
for i, s in enumerate(sentences):
    sentences[i] =  f"{s}.".strip()

num_sentences = len(sentences)

# ***
# get cleaned version of each sentence
cleaned_sentences: List[str] = [" ".join(_tokenize_text(s)) for s in sentences]  # list of sentences
corpus: List[str] = [" ".join(cleaned_sentences)]  # all sentences as one document


# ****************************************************
#
# build similarity matrix
#
# ****************************************************
# ***
# fit vectorizer
sentence_vectors = VECTORIZER.fit_transform(cleaned_sentences)  # see top comment: sentences vs. corpus of one doc

# ***
# initialize similarity matrix with dimension (n, n) (n = number of sentences)
similarity_matrix = np.zeros([num_sentences, num_sentences])

# ***
# calculate cosine similarity between pairs of sentences
for i in range(num_sentences):
    for j in range(num_sentences):
        if i == j:
            continue
        similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i], sentence_vectors[j])[0,0]


#****************************************************
#
# get scores 
# compare with
# power_method(matrix, epsilon) in https://github.com/miso-belica/sumy/blob/master/sumy/summarizers/lex_rank.py
#
# ****************************************************
epsilon = 0.1
lambda_val = 1.0
scores = np.array([1.0 / num_sentences] * num_sentences)  # p_vector

while lambda_val > epsilon:
    next_p = np.dot(similarity_matrix, scores)
    lambda_val = np.linalg.norm(np.subtract(next_p, scores))
    scores = next_p
    

# ****************************************************
#
# Summary Extraction
#
# ****************************************************
ranked_sentences: List[Tuple[float, str]] = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

# ***
# get the n % most relevant sentences
percentage_text_reduction = 30
num_sentences = int((percentage_text_reduction * len(sentences))/100)

print(f"----- {num_sentences} most relevant sentences ({percentage_text_reduction}%) -----")
for i in range(num_sentences):
    sentence = ranked_sentences[i][1]
    print(f"{i+1} - {sentence}")
