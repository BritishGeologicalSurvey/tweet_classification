#-*- coding: utf-8 -*-
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

from classification_module import NLTKPreprocessor,identity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.metrics import accuracy_score as acc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as tts

# Notes1.- Other is positive(1) - Aurora is negative(0)

# Notes2.- The underlying C implementation uses a random number generator to select features when fitting the model. It is thus not uncommon to have slightly different results for the same input data

def normalize(text):
   # text=text.encode('ascii','ignore')
    text=text.lower()
    punct=';,?$&!~<>)('
    for sym in punct:
        text=text.replace(sym,' ')
    temp_corpus=text.split(' ')
    stops=['a','an','is','rt','via','the','im','u','num','of','it','as']
    text=[token for token in temp_corpus if token not in stops]
    text=' '.join(text)
    return text

def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper




@timeit
def build_and_evaluate(X, y, outpath=None, verbose=True):
    """
    Builds a classifer for the given list of documents and targets in two
    stages: the first does a train/test split and prints a classifier report,
    the second rebuilds the model on the entire corpus and returns it for
    operationalization.
    X: a list or iterable of raw strings, each representing a document.
    y: a list or iterable of labels, which will be label encoded.
    Can specify the classifier to build with: if a class is specified then
    this will build the model with the Scikit-Learn defaults, if an instance
    is given, then it will be used directly in the build pipeline.
    If outpath is given, this function will write the model as a pickle.
    If verbose, this function will print out information to the command line.
    """
    @timeit
    def build(Classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        if Classifier=="NB":
            classifier = MultinomialNB()
        elif Classifier=="SVC":
            classifier = SVC()
        elif Classifier=="LSVC":
            classifier = LinearSVC()
        elif Classifier=="LR":
            classifier = LogisticRegression()
        elif Classifier == "NN":
             classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1) 
        else:
            classifier = SGDClassifier()
        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity,preprocessor=None, lowercase=False, min_df=0.01, max_df=0.95)),
            ('classifier', classifier),
        ])
        model.fit(X, y)
        return model

    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y)

    # Begin evaluation
    if verbose: print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
   # pdb.set_trace()
    print("Train size %s" % len(X_train))
    print("Y train size %s" % len(y_train))
    model, secs = build("NN", X_train, y_train)
    #model, secs = build("KK", X_train, y_train)

    if verbose: print("Evaluation model fit in {:0.3f} seconds".format(secs))
    if verbose: print("Classification Report:\n")

    y_pred = model.predict(X_test)
    tot=0	
    for l in range(len(y_pred)):
        if y_pred[l]!=y_test[l]:
            tot=tot+1
    print("Number of test %s, numer of errors %s" % (len(y_pred), tot))	
    print(clsr(y_test, y_pred, target_names=labels.classes_))
    accuracy = acc(y_test,y_pred)
    print('Accuracy: {}'.format(accuracy))


	 
    if verbose: print("Building complete model and saving ...")
    print("size of the total data is %s and total label %s" % (len(X),len(y)))
    model, secs = build("NN", X, y)
    #model, secs = build("KK", X, y)
    model.labels_ = labels
    if verbose: print("Complete model fit in {:0.3f} seconds".format(secs))

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


def show_most_informative_features(model, text=None, n=20):
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.
    Note that this function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']
    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {} model.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append("Classified as: {}".format(model.predict([text])))
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
        )

    return "\n".join(output)

if __name__ == "__main__":
    # Set path to save model
    PATH = "./model.pickle"
    #Set path/names of training files
    filename_list = ['train_BGS.pos','train_BGS.neg']   
    array = []
    labels = []
    with open(filename_list[0], "r") as pos, open(filename_list[1]) as neg:
            for line in pos:
                    array.append(normalize(line))
                    labels.append('Aurora')
            for line in neg:
                    array.append(normalize(line))
                    labels.append('Other')
    model = build_and_evaluate(array,labels,outpath=PATH)
