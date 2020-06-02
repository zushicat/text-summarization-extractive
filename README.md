# text-summarization
Create summaries of text documents.

### Extractive (unsupervised)
- extractive_simple.py    
  Get most relevant sentences with tf-idf metrics
- extractive_text_rank.py    
  TextRank implementation (based on word vectors)
- extractive_lex_rank.py
  LexRank implementation (cosine similarity on tf-idf metrics)



Download pre-trained word vectors glove.6B.zip, unzip file and place glove.6B.100d.txt in /data directory:    
https://nlp.stanford.edu/projects/glove/


## Further Reading
### General
- "Wikipedia: Automatic summarization": https://en.wikipedia.org/wiki/Automatic_summarization
- "Automatic Text Summarization with Machine Learning — An overview": https://medium.com/luisfredgs/automatic-text-summarization-with-machine-learning-an-overview-68ded5717a25
- "A Quick Introduction to Text Summarization in Machine Learning": https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f
- "An Overview of Summarization": https://blog.agolo.com/an-overview-of-summarization-551499a144d5
- "A Gentle Introduction to Text Summarization in Machine Learning": https://blog.floydhub.com/gentle-introduction-to-text-summarization-in-machine-learning/
- "Text Summarization in Python: Extractive vs. Abstractive techniques revisited": https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/

### Packages
- Graphs / Networks
    - "networkx": http://networkx.github.io/

### Extractive
- Gensim Summarize
    - "Tutorial: automatic summarization using Gensim": https://rare-technologies.com/text-summarization-with-gensim/
- TextRank
    - "An Introduction to Text Summarization using the TextRank Algorithm (with Python implementation)": https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/
- LexRank
    - "LexRank method for Text Summarization": https://iq.opengenus.org/lexrank-text-summarization/
    - "LexRank: Graph-based Lexical Centrality as Salience in Text Summarization": https://www.aaai.org/Papers/JAIR/Vol22/JAIR-2214.pdf