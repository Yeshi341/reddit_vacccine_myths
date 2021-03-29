import numpy as np
import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# function to clean
def clean_data(text):
    '''
    This function takes in a text string argument and 
    removes all unwanted objects from it, thereby cleaning the text
    '''
    # remove text within brackets
    text = re.sub('\[.*?\]', '', text)
    
    # remove text within parenthesis
    text = re.sub('\(.*?\)', '', text)
    
    # remove numbers
    text = re.sub('\w*\d\w*', '', text)
    
    # remove whitespaces
#     text = re.sub('\s+', ' ', text)
    
    # remove any quotes
    text = re.sub('\"+', '', text)
    
    # remove punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # lowercase all text
    text = text.lower()
    
    # remove anything not a-z
#     text = re.sub('[^a-z]', '', text)
        
    return text



stop_words = set(stopwords.words('english'))
#tokenize and remove any stop words
def tokenize(text):
    ''' tokenizes text passed in the function argument'''
    
    tokens = nltk.word_tokenize(text)
    # remove any stop words in the text
    stopwords_removed = [token for token in tokens if token not in stop_words]
    return stopwords_removed


def stemming(tokenized_text):
    ''' removes morphological affixes from words in text, leaving only the word stem'''
    
    stemmer = PorterStemmer()
    # remove any stop words in the text
    stemmed = [stemmer.stem(word) for word in tokenized_text]
    return ' '.join(stemmed)

def find_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    polarity_score = sid.polarity_scores(text)['compound']
    
    if polarity_score < 0:
        sentiment = "negative"
    elif polarity_score > 0:
        sentiment = "positive"
    elif polarity_score == 0:
        sentiment = 'neutral'
    return sentiment

# function to return a count vectorized representation of text as a dictionary
def count_vectorize(text, vocab=None):
# vocab is an optional parameter set to default to None
# vocab is just in case we use a vocabulary that contains words not seen in the song

    if vocab:
        unique_words = vocab
    else:
        unique_words = list(set(text))
    
    # empty dictionary with keys as the unique words in the corpus
    text_dict = {i:0 for i in unique_words}
    
    # adding count values for each unique words
    for t in text:
        text_dict[t] +=1
    
    return text_dict