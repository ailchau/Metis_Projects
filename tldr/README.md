# TL;DR (Too long; didn't read)

The goal of this project was to generate summaries of new articles using natural language process and deep learning. I identified crime-related news articles by clustering [CNN and Dailymail](http://cs.nyu.edu/~kcho/DMQA/) news articles based on their underlying topics. A sequence-to-sequence model, involving LSTM layers and an attention mechanism, was used to train the crime news summary generating model. The Google News [Word2Vec](https://code.google.com/archive/p/word2vec/) word embeddings were also used.

Tools: spaCy, Scikit-learn, Tensorflow, and AWS

### Main contents:
- [gather_articles.py](gather_articles.py) gathered and compiled the articles and summaries from separate files
- [clean_text.py](clean_text.py) cleaned and standardized the articles and summaries
- [topic_modeling_clustering.ipynb](topic_modeling_clustering.ipynb) identified latent topics and clustered the articles based on their underlying topics
- [prepare_text.ipynb](prepare_text.ipynb) prepared the articles and summaries (as well as necessary information) 
- [train_model.py](train_model.py) built and trained the summary generating model using the train articles
- [generate_summaries.py](generate_summaries.py) generated summaries (2-3 sentences long) using test articles
- [tldr_presentation.pdf](tldr_presentation.pdf): presentation slides
