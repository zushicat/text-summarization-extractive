'''
TextRank

see also:
"An Introduction to Text Summarization using the TextRank Algorithm (with Python implementation)":
https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/

Download pre-trained word vectors (glove.6B.100d.txt): https://nlp.stanford.edu/projects/glove/

----- 6 most relevant sentences (30%) -----
1 - At home, he oversaw the legal union of England and Wales with the Laws in Wales Acts 1535 and 1542, 
    and he was the first English monarch to rule as King of Ireland following the Crown of Ireland Act 1542.
2 - Henry's contemporaries considered him an attractive, educated, and accomplished king.
3 - Despite the money from these sources, he was continually on the verge of financial ruin due to his personal extravagance, 
    as well as his numerous costly and largely unsuccessful wars, particularly with King Francis I of France, 
    Holy Roman Emperor Charles V, James V of Scotland and the Scottish regency under the Earl of Arran and Mary of Guise.
4 - Henry VIII was King of England from 1509 until his death in 1547.
5 - His disagreement with Pope Clement VII on the question of such an annulment led Henry to initiate the English Reformation, 
    separating the Church of England from papal authority.
6 - Henry is best known for his six marriages, and, in particular, his efforts to have his first marriage annulled.

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


TOKENIZER = RegexpTokenizer(r"\w+")
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))


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
loaded_text = " ".join(loaded_text.split())  # remove double space


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
corpus: str = " ".join(cleaned_sentences)  # all sentences as one document


# ****************************************************
#
# word vectors
#
# ****************************************************
# ***
# Extract word vectors from file (a little time consuming)
word_embeddings: Dict[str, np.ndarray] = {}

with open("data/glove.6B.100d.txt") as f:
    for line in f:
        values = line.split()
        token = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[token] = coefs

# ***
# apply word vectors on sentences
sentence_vectors: List[np.ndarray] = []

for sentence in cleaned_sentences:
    if len(sentence) == 0:
        vectors = np.zeros((100,))  # fill with zeros
    else:
        vectors = sum([
            word_embeddings.get(token, np.zeros((100,))) for token in sentence.split()  # get vectors of token or fill with zero
        ])/(len(sentence.split())+0.001)
    
    sentence_vectors.append(vectors)


# ****************************************************
#
# similarity matrix
#
# ****************************************************
# ***
# initialize similarity matrix with dimension (n, n) (n = number of senteces)
similarity_matrix = np.zeros([num_sentences, num_sentences])

# ***
# calculate cosine similarity between pairs of sentences
for i in range(num_sentences):
    for j in range(num_sentences):
        if i == j:
            continue
        similarity_matrix[i][j] = cosine_similarity(
            sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100)
        )[0,0]


# ****************************************************
#
# text rank scores from graph
#
# ****************************************************
graph = nx.from_numpy_array(similarity_matrix)
scores: Dict[int, float] = nx.pagerank(graph)

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
