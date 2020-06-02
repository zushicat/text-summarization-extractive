'''
Get the n (i.e. 5) most relevant sentences from a document.

Based on:
"Automatic Extractive Text Summarization using TF-IDF":
https://medium.com/voice-tech-podcast/automatic-extractive-text-summarization-using-tfidf-3fc9a7b26f5

To be discussed: 
Vectorize on sentences or on whole corpus (of this one document)? 

Sorted results of the 5 most relevant sentences:
- on sentences
    1 - He was an author and composer.
    2 - He was succeeded by his son Edward VI.
    3 - He also greatly expanded royal power during his reign.
    4 - He has been described as "one of the most charismatic rulers to sit on the English throne".
    5 - Henry VIII was King of England from 1509 until his death in 1547.
- on corpus
    1 - He was an author and composer.
    2 - He was succeeded by his son Edward VI.
    3 - He also greatly expanded royal power during his reign.
    4 - Henry VIII was King of England from 1509 until his death in 1547.
    5 - Henry's contemporaries considered him an attractive, educated, and accomplished king.
'''

import json
import re
from typing import List, Tuple

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


TOKENIZER = RegexpTokenizer(r"\w+")
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

VECTORIZER = TfidfVectorizer()


def _tokenize_text(text:str) -> List[str]:
    def filter_by_pos(token: List[str]) -> List[str]:  # to be discussed: is it helpful in this context?
        return [t for t, pos in pos_tag(token) if pos in ["NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]
    def remove_stopwords(token: List[str]) -> List[str]:
        return [w for w in token if not w in STOPWORDS]  # filtered token
    def tokenize_text(text: str) -> List[str]:
        return TOKENIZER.tokenize(text)  # list of token without punctuation etc.
    def get_token_lemmata(token: List[str]) -> List[str]:
        return [LEMMATIZER.lemmatize(t) for t in token]
    def get_token_stems(token: List[str]) -> List[str]:
        return [STEMMER.stem(t) for t in token]
    
    token: List[str] = get_token_stems(remove_stopwords(tokenize_text(text.lower())))
    # token = filter_by_pos(token)
    
    return token


# ***
#
with open("data/source_texts/wikipedia_a_1.txt") as f:
    loaded_text: str = f.read()

loaded_text = loaded_text.replace("\n\n", "")

# ***
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

# ***
# get cleaned version of each sentence
cleaned_sentences: List[str] = [" ".join(_tokenize_text(s)) for s in sentences]  # list of sentences
corpus: List[str] = [" ".join(cleaned_sentences)]  # all sentences as one document

# ***
# fit vectorizer
VECTORIZER.fit(corpus)  # see top comment: sentences vs. corpus of one doc

# ***
# get vectors of token from transformed sentences (normalized with length of cleaned sentence)
sentence_values: List[float] = []
for sentence in cleaned_sentences:
    num_token = len(sentence.split())
    if num_token == 0:  # ignore empty sentences
        sentence_values.append(0.0)
        continue

    token_values = VECTORIZER.transform([sentence])
    sentence_values.append(sum(token_values.data)/num_token)  # additionally normalize sum with num token in cleaned sentence

# ***
# sort by values: result is list of sorted tuples [(i, val), ...]
sorted_by_values: List[Tuple[int, float]] = [i for i in sorted(enumerate(sentence_values), key=lambda x:x[1], reverse=True)]   

# ***
# get the n % most relevant sentences
percentage_text_reduction = 30
num_sentences = int((percentage_text_reduction * len(sentences))/100)

print(f"----- {num_sentences} most relevant sentences ({percentage_text_reduction}%) -----")
for i, vals in enumerate(sorted_by_values[:num_sentences]):
    sentence = sentences[vals[0]]
    print(f"{i} - {sentence}")
