import os
import re
import numpy as np
import pandas as pd
import random
import pickle
import spacy
from nltk.corpus import stopwords

# load English models
nlp = spacy.load('en')

# load articles and summaries
articles_list = pickle.load(open("articles_list.pkl", "rb"))

# ---------------------------

# helper functions to clean and standardize articles and summaries

# list of contractions taken from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


def remove_contractions(text):
    """
    Replace contractions with their complete form.
    """
    text = text.split()
    new_text = []
    
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
            
    return " ".join(new_text)  


def clean_text(text):
    """
    Clean up and standardize text. 
    text should be a single string of text (articles or highlights)
    """
    # replace contractions with both words
    text = remove_contractions(text)

    # standardize possibly relevant terms
    text = re.sub(r"^[0-9]{2}:[0-9]{2}$", "time", text)
    text = re.sub(r"^[0-9]{1}:[0-9]{2}$", "time", text)    
    text = re.sub(r"^[\d]+[.:][\d]+[Aa][Mm]$", "time", text)
    text = re.sub(r"^[\d]+[.:][\d]+[Pp][Mm]$", "time", text)
    text = re.sub(r"[\d]{4}", "year", text)
    text = re.sub(r"[\w\d\./:]+\.com", " website ", text)
    text = re.sub(r"@@[\S]+", "handle", text)
    text = re.sub(r"\%", " percent", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"/", " ", text)
    
    # remove punctuation and unnecessary characters
    text = re.sub(r"[_\'\"\-;%()|+&=*%.,!?:#\$@\[\]/\\]", "", text)

    # remove stopwords
    remain = ""
    doc = nlp(text)
    for token in doc:
        if token.is_stop:
            continue
        elif token.is_space:
            continue
        elif token.is_punct:
            continue
        elif len(token)<=1:
            continue
        else:
            remain += str(token) + " "
            
    return remain


def truncate_text(text, number_of_words):
    """
    Truncate length of text.
    number_of_words should be an integer
    """
    return " ".join(text.split()[:number_of_words])


def clean_highlights(text):
    """
    Clean highlights/summary text.
    """
    text = remove_contractions(text)

    # standardize possibly relevant terms
    text = re.sub(r"^[0-9]{2}:[0-9]{2}$", "time", text)
    text = re.sub(r"^[0-9]{1}:[0-9]{2}$", "time", text)    
    text = re.sub(r"^[\d]+[.:][\d]+[Aa][Mm]$", "time", text)
    text = re.sub(r"^[\d]+[.:][\d]+[Pp][Mm]$", "time", text)
    text = re.sub(r"[\d]{4}", "year", text)
    text = re.sub(r"[\w\d\./:]+\.com", " website ", text)
    text = re.sub(r"@@[\S]+", "handle", text)
    text = re.sub(r"\%", " percent", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"/", " ", text)
    
    # remove punctuation and unnecessary characters
    text = re.sub(r"[_\'\"\-;%()|+&=*%.,!?:#\$@\[\]/\\]", "", text)
    
    return text


# keep only cleaned articles that have less than 500, and truncate article length
def clean_truncate(dict_articles, article_length=500, truncate_length=150):
    """
    Combine cleaning and truncating functions.
    dict_articles should be a list of dictionaries containing article and highlights
    """
    cleaned_articles = []
    
    for index, article in enumerate(dict_articles):

        cleaned = clean_text(article["article"])
        
        if len(cleaned.split())<=article_length:
            cleaned_dict = {}
            cleaned_dict["article"] = truncate_text(cleaned,truncate_length)
            cleaned_dict["highlights"] = clean_highlights(article["highlights"])
            cleaned_dict["file"] = article["file"]
            cleaned_articles.append(cleaned_dict)

        if index%1000==0:
            print("Cleaning. Currently at {}".format(index))
            
    print("Number of cleaned articles: {}".format(len(cleaned_articles)))
    
    return cleaned_articles


# ---------------------------

# clean and truncate articles and summaries

cleaned_articles = clean_truncate(articles_list)

pickle.dump(cleaned_articles, open("cleaned_articles_dict.pkl", "wb"))


# create train and test set
# num_training = 0.8

# random.seed(910)
# random.shuffle(cleaned_articles)
# index_split = int(len(cleaned_articles)*num_training)
# train_articles = cleaned_articles[:index_split]
# test_articles = cleaned_articles[index_split:]

# pickle.dump(train_articles, open("train_articles_dict.pkl", "wb"))
# pickle.dump(test_articles, open("test_articles_dict.pkl", "wb"))

# print("Number of train articles: {}".format(len(train_articles)))
# print("Number of test articles: {}".format(len(test_articles)))
