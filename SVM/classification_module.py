import os
import time
import string
import pickle
import json
import re
from operator import itemgetter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as tts


# Notes1.- Other is positive(1) - Aurora is negative(0)

# Notes2.- The underlying C implementation uses a random number generator to select features when fitting the model. It is thus not uncommon to have slightly different results for the same input data

http_re = re.compile(r'\s+http://[^\s]*')
remove_ellipsis_re = re.compile(r'\.\.\.')
at_sign_re = re.compile(r'\@\S+')
punct_re = re.compile(r"[\"'\[\],.:;()\-&!]")
price_re = re.compile(r"\d+\.\d\d")
number_re = re.compile(r"\d+")

# converts to lower case and clean up the text
def normalize_tweet(tweet):
    t = tweet.lower()
    #t = re.sub(price_re, 'PRICE', t)
    t = re.sub(remove_ellipsis_re, '', t)
    t = re.sub(http_re, ' LINK', t)
    t = re.sub(punct_re, '', t)
    t = re.sub(at_sign_re, '@', t)
    t = re.sub(number_re, 'NUM', t)
    #t = re.sub(r"\n", " ", t) # remove newlines from text
    t = re.sub(r'[^\w]', ' ', t)
    return t


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.
    """

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        """
        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.
        """
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct      = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, X):
        """
        Actually runs the preprocessing on each document.
        """
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """
        # Break the document into sentences
	#print("Document is %s" % document)

        try:
            sents = sent_tokenize(document)
        except ValueError:
            sents = "ERROR"
        if sents!= "ERROR":
            for sent in sents:
                try:
                    sent = normalize_tweet(sent)
                    for token, tag in pos_tag(wordpunct_tokenize(sent)):
                        token = token.lower() if self.lower else token
                        token = token.strip() if self.strip else token
                        token = token.strip('_') if self.strip else token
                        token = token.strip('*') if self.strip else token
                        token = token.rstrip('?:!.,;"!@')
                        if token in self.stopwords or all(char in self.punct for char in token):
                            continue
                        lemma = self.lemmatize(token, tag)
                        yield lemma
                except ValueError:
                    pass

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


