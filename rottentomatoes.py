# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 21:48:12 2019

@author: Eunice
"""

import numpy as np            
import pandas as pd


""" READING THE DATA """
print("#### Loading the data\t\t####")
df = pd.read_csv('rotten_tomatoes_reviews.csv', header=0)
df.info()


""" TRANSFORM THE LABELS """
print("\n#### Preprocessing\t\t####")
def transform(s):
    """ from string to number
          rotten -> 0
          fresh -> 1
    """
    d = { 'rotten':0, 'fresh':1 }
    return d[s]

df['Freshness'] = df['Freshness'].map(transform)
labels = ['rotten', 'fresh']

X_all_df = df[ 'Review' ]
y_all_df = df[ 'Freshness' ]

X_all = X_all_df.values
y_all = y_all_df.values 

#print(X_all[:5])
#print(y_all[:5])


""" LOWERCASE AND REMOVE CHARACTERS """
import re

REPLACE_NO_SPACE = re.compile("[;:\'\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

clean_X = preprocess_reviews(X_all)
#print(clean_X[0:5])
#print(y_all[0:5])


""" LEMMATIZER """
import nltk
import string
from nltk.stem import WordNetLemmatizer

lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


""" SENTIWORDNET """
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None 


from sklearn.base import BaseEstimator, TransformerMixin

# Sentiscore Transformer
class SentiScoreExtractor(BaseEstimator, TransformerMixin):
    """Takes in data, outputs SentiScores"""

    def __init__(self):
        pass
    
    def review_polarity(self, review):
        """ Return a sentiment score of the review which I want to use as a feature """
        """ However, I am not sure how to add these to my features since I also use a Vectorizer """
        sentiment = 0.0
        tokens_count = 0 
        
        words_tagged = nltk.pos_tag(nltk.word_tokenize(review))
        
        for word, tag in words_tagged:
                wn_tag = penn_to_wn(tag)
                if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
                    continue
     
                lemma = lemmer.lemmatize(word, pos=wn_tag)
                if not lemma:
                    continue
     
                synsets = wn.synsets(lemma, pos=wn_tag)
                if not synsets:
                    continue
     
                # Take the first synset, the most common
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
     
                # Get the positive and negative score (which are both floats>0)
                # Get the sentiment score by subtracting the neg_score from the pos_score
                sentiment += swn_synset.pos_score() - swn_synset.neg_score()
                #print(word, swn_synset.pos_score(), swn_synset.neg_score())
                tokens_count += 1 # Do I want to divide by the total number of tokens?
                                  # tokens = words that had sentiment scores
#        if sentiment == 0.0:
#            return 0
#        elif sentiment/tokens_count > 0.33:
#            return 1
#        elif sentiment/tokens_count < -0.33:
#            return -1
#        else:
#            return 0
        return sentiment

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        sentiscores = []
        for review in X:
            sentiscores.append(self.review_polarity(review))
        return sentiscores

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


print("\n#### FEATURE ENGINEERING\t####")
""" VECTORIZER """
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = ['in', 'of', 'at', 'a', 'an', 'the', 'and', 'it']

TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, ngram_range=(1,3), stop_words=stop_words)
#TfidfVec.fit(clean_X)
#X = TfidfVec.transform(clean_X)


""" COMBINING FEATURES """

from sklearn.pipeline import Pipeline, FeatureUnion

# Need to cast Sentiscore as transposed matrix
class ArrayCaster(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, data):
    return np.transpose(np.matrix(data))

union = FeatureUnion([("tdidf", TfidfVec),
                      ('senti data', Pipeline([
                            ("senti", SentiScoreExtractor()),
                            ('caster', ArrayCaster())
                          ]))])
    
    
union.fit(clean_X)  
X = union.transform(clean_X)


""" SPLIT INTO TRAINING AND TEST SETS """
print("\n#### Data Partitioning\t####")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_all, test_size=0.1, stratify=y_all)

#print(X_train[:5])
#print(y_train[:5])


""" MODELING """
print("\n#### Building the model\t####")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

#split training set further into train and validation sets
#X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.22, stratify=y_train)

print("\n")
""" Choosing the best C """
#for c in [0.01, 0.05, 0.25, 0.5, 1]:
#    lr = LogisticRegression(C=c)
#    lr.fit(X_train2, y_train2)
#    svm = LinearSVC(C=c)
#    svm.fit(X_train2, y_train2)
#    print ("LogReg: Accuracy for C=%s: %s" 
#           % (c, accuracy_score(y_val, lr.predict(X_val))))
#    print ("SVM: Accuracy for C=%s: %s" 
#           % (c, accuracy_score(y_val, svm.predict(X_val))))

logreg_model = LogisticRegression(C=1) #C=1 led to highest accuracy score
logreg_model.fit(X_train, y_train)
logreg_accuracy = accuracy_score(y_test, logreg_model.predict(X_test))
print("\n")
print ("LogReg Accuracy: %s" 
       % logreg_accuracy)
print("LogReg F1-score: %s" % f1_score(y_test, logreg_model.predict(X_test)))
print("\n")

svm_model = LinearSVC(C=1) #C=1 led to highest accuracy score
svm_model.fit(X_train, y_train)
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
print("\n")
print ("SVM Accuracy: %s" 
       % svm_accuracy)
print("SVM F1-score: %s" % f1_score(y_test, svm_model.predict(X_test)))
print("\n")


#final model to predict Avengers and Aladdin reviews
final_model = logreg_model 
if svm_accuracy > logreg_accuracy:
    final_model = svm_model
    print("final model is SVM\n") 
else:
    print("final model is Logistic Regression\n")


""" TESTING AVENGERS:ENDGAME REVIEWS """

print("Testing Avengers: Endgame reviews")
endgame = ["What you will be getting when you walk into an inevitably overstuffed movie theater is something singular that reflects our age in a way that none of the MCU films that preceded it have-indeed, very few Hollywood spectacles ever have.", 
           "The MCU will go on and on, but this chapter - and the American pragmatism vs. American ideals bromance that drove it - have well and truly come to their \"Excelsior! Nuff said!\" moment.",
           "What's missing from \"Endgame\" is the free play of imagination, the liberation of speculation, the meandering paths and loose ends that start in logic and lead to wonder.",
           "Endgame consists almost entirely of the downtime scenes that were always secretly everyone's favorite parts of these movies anyway.",
           "There are horror movies and political dramas with premises less nightmarish than the one in Avengers: Endgame... This is an interesting approach for a superhero film to take.",
           "You can easily duck out during the middle hour, do some shopping, and slip back into your seat for the climax. You won't have missed a thing."]
endgame_actual = ['fresh', 'fresh', 'rotten', 'fresh', 'fresh', 'rotten']
endgame = preprocess_reviews(endgame)
endgame_v = union.transform(endgame)
#endgame_v = TfidfVec.transform(endgame)
endgame_p = final_model.predict(endgame_v)

print("\n")
# the headers
s = "{0:<11} | {1:<11}".format("Predicted","Actual")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)

# the separators
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)

# here is the table...
for pi, ai in zip( endgame_p, endgame_actual ):
    # pi and ai are the index of the predicted and actual label
    plabel = labels[pi]  # tn is target_names - see above
    alabel = ai #see endgame_actual 
    s = "{0:<11} | {1:<11}".format(plabel,alabel)
    print(s)

print("\n\n")

""" TESTING ALADDIN REVIEWS """

print("Testing Aladdin reviews")
aladdin = ["Marwan Kenzari snarls it up as the villainous Jafar, while Nasim Pedrad is endearing and funny as Jasmine's handmaiden and best friend Dalia...",
           "The new telling brings little that feels essential, and the missing elements-chiefly, the controlled chaos and unbridled comedy of the late Robin Williams' vocal performance-have given the film's candy-colored visual palate a homesick pall.",
           "In short, it's a whole old world.",
           "There are efforts made, whether through good faith or just market savvy, to update Princess Jasmine into a people's champion who might prefer ruling to romance. Enough to make you wish the Disney people had gone whole hog and just called it Jasmine.",
           "With Aladdin, they've done the leveling with just enough style and pizazz that most moviegoers won't care that it's a retread, and the leads are good enough to make you hope they'll go on to something real.",
           "Smith understandably didn't want to compete with Williams, but as the big, blue, top-knotted Genie, he's uncharacteristically bland. Even the magic carpet in this movie looks bummed out."]
aladdin_actual = ['fresh', 'rotten', 'rotten', 'rotten', 'fresh', 'rotten']
aladdin = preprocess_reviews(aladdin)
aladdin_v = union.transform(aladdin)
#aladdin_v = TfidfVec.transform(aladdin)
aladdin_p = final_model.predict(aladdin_v)

print("\n")
# the headers
s = "{0:<11} | {1:<11}".format("Predicted","Actual")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)

# the separators
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)

# here is the table...
for pi, ai in zip( aladdin_p, aladdin_actual ):
    # pi and ai are the index of the predicted and actual label
    plabel = labels[pi]  # tn is target_names - see above
    alabel = ai
    s = "{0:<11} | {1:<11}".format(plabel,alabel)
    print(s)

print("\n\n")


""" LOOKING AT TOP N-GRAMS """

from collections import defaultdict
features_by_gram = defaultdict(list)
for f, w in zip(TfidfVec.get_feature_names(), final_model.coef_[0]):
    features_by_gram[len(f.split(' '))].append((f, w))
    
top_n = 5

# POSITIVE N-GRAMS
for gram, features in features_by_gram.items():
    top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
    print('{}-gram Best Positive:'.format(gram))
    for feat in top_features:
        print(feat)
    print("\n")
    
# NEGATIVE N-GRAMS
for gram, features in features_by_gram.items():
    top_features = sorted(features, key=lambda x: x[1])[:top_n]
    print('{}-gram Best Negative:'.format(gram))
    for feat in top_features:
        print(feat)
    print("\n")