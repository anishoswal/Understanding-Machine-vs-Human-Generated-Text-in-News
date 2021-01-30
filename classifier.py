import numpy as np
import pandas as pd
import pickle
import nltk
import nltk.corpus 
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import  LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sb

#read the csv data provided to us as the tweepfake dataset
train_tweets, test_tweets = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")


#BAG OF WORDS TECHNIQUE
#Finding the count vector for all the provided tweets in the training dataset
countVectorz = CountVectorizer()
counterTrain = countVectorz.fit_transform(train_tweets['text'].values)

#find and make the tf-df frequency features using the tfidf transformer
tfidfVect = TfidfTransformer()
tfidfTrainz = tfidfVect.fit_transform(counterTrain) #not used

#create the various types of features which can be used later by the different classification algorithms for classification
def Features(tweettext, ind):
    return {'word': tweettext[ind],
        'is_first': ind == 0,
        'is_last': ind == len(tweettext) - 1,
        'is_capitalized': tweettext[ind][0].upper() == tweettext[ind][0],
        'is_all_caps': tweettext[ind].upper() == tweettext[ind],
        'is_all_lower': tweettext[ind].lower() == tweettext[ind],
        'prefix-1': tweettext[ind][0],
        'prefix-2': tweettext[ind][:2],
        'prefix-3': tweettext[ind][:3],
        'suffix-1': tweettext[ind][-1],
        'suffix-2': tweettext[ind][-2:],
        'suffix-3': tweettext[ind][-3:],
        'prev_word': '' if ind == 0 else tweettext[ind - 1],
        'next_word': '' if ind == len(tweettext) - 1 else tweettext[ind + 1],
        'has_hyphen': '-' in tweettext[ind],
        'is_numeric': tweettext[ind].isdigit(),
        'capitals_inside': tweettext[ind][1:].lower() != tweettext[ind][1:]}

#finding the n-gram tfidf vectorizer  
ngramTFIDF = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)

#Using Word2Vec and the text glove file
with open("glove.6B.50d.txt", "rb") as data:
    word2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in data}

#first we implement the bag of words technique for 3 different classifiers

#logistic regression
pLogR = Pipeline([('countLOGR',countVectorz),('classifierLOGR',LogisticRegression(max_iter=10000))])
#pLogR.fit(train_tweets['text'],train_tweets['class_type'])

#linear SVM classfier
pSVM = Pipeline([('countSVM',countVectorz),('classifierSVM',svm.LinearSVC(max_iter=10000))])
#pSVM.fit(train_tweets['text'],train_tweets['class_type'])

#random forest
pRandomForest = Pipeline([('countRF',countVectorz),('classifierRF',RandomForestClassifier(n_estimators=200,n_jobs=3))])
#pRandomForest.fit(train_tweets['text'],train_tweets['class_type'])

def results(classifier):
    train_text = train_tweets['text']
    train_y = train_tweets['class_type']
    test_text = test_tweets['text']
    test_y = test_tweets['class_type']

    classifier.fit(train_text, train_y)
    predictions = classifier.predict(test_text)
    print(f1_score(test_y, predictions, pos_label="human"))
    print(confusion_matrix(test_y, predictions))
    print(classification_report(test_y, predictions))
print("======================================================BAG OF WORDS============================================")
print("==========================================================LogR================================================")
results(pLogR)
#print(classification_report(test_tweets['class_type'], predictLogR))
print("==============================================================================================================")
print("==========================================================SVM=================================================")
results(pSVM)
#print(classification_report(test_tweets['class_type'], predictSVM))
print("==============================================================================================================")
print("==========================================================RF==================================================")
results(pRandomForest)
#print(classification_report(test_tweets['class_type'], predictRandomForest))
print("==============================================================================================================")
print("==============================================================================================================")

#NGRAM TECHNIQUE
#then we implement the ngrams technique for the same 3 classifiers

#logistic regression classifier
ngramLogR = Pipeline([('ngramLOGR',ngramTFIDF),('classifierLOGR',LogisticRegression(penalty="l2",C=1))])
#ngramLogR.fit(train_tweets['text'],train_tweets['class_type'])

#linear SVM classifier
ngramSVM = Pipeline([('ngramSVM',ngramTFIDF),('classifierSVM',svm.LinearSVC())])
#ngramSVM.fit(train_tweets['text'],train_tweets['class_type'])

#random forest classifier
ngramRandForest = Pipeline([('ngramRF',ngramTFIDF),('classifierRF',RandomForestClassifier(n_estimators=300,n_jobs=3))])
#ngramRandForest.fit(train_tweets['text'],train_tweets['class_type'])

print("\n\n")
print("=========================================================N-GRAMS==============================================")
print("==========================================================LogR================================================")
results(ngramLogR)
#print(classification_report(test_tweets['class_type'], predictNgramLogR))
print("==============================================================================================================")
print("==========================================================SVM=================================================")
results(ngramSVM)
#print(classification_report(test_tweets['class_type'], predictNgramSVM))
print("==============================================================================================================")
print("==========================================================RF==================================================")
results(ngramRandForest)
#print(classification_report(test_tweets['class_type'], predictNgramRF))
print("==============================================================================================================")
print("==============================================================================================================")

"""
Out of all the different classifiers used we take into account the best two performing models and then using the F1 scores
we try to do parameter optimization. Here, we see that one of the best models which we can use is logistic regression using
bag of words and also use the random forest classifier for n gram as it performs better on our Project 1 data. These are the ones which
perform well are the ones we chose to optimize the parameters using grid search algorithm.
"""





"""
NOTE: The grid search algorithm uses a lot of memory and will crash python if used on a computer which does not have enough
memory to allocate so we used a high performance computer to generate the results from the grid search algorithm.
"""

print("\nNOTE: The grid search algorithm uses a lot of memory and will crash python if used on a computer which does not have enough memory to allocate so we used a high performance computer to generate the results from the grid search algorithm and hence the code for grid search is commented out.")







#GRID SEARCH PARAMETER OPTIMIZATION
#random forest
"""
gridclassify = GridSearchCV(ngramRandForest, {'tfidf_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],'tfidf_useidf': (True, False),'tfidf_depth': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)}, n_jobs=-1)
gridclassify = gridclassify.fit(train_tweets['text'][:10000],train_tweets['class_type'][:10000])
gridclassify.best_score_
gridclassify.best_params_
gridclassify.cv_results_
"""
#logistic regression parameters
"""
gridclassify = GridSearchCV(pLogR, {'logr_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],'rogr_idf': (True, False),'logr_smooth': (True, False)}, n_jobs=-1)
gridclassify = gridclassify.fit(train_tweets['text'][:10000],train_tweets['class_type'][:10000])
gridclassify.best_score_
gridclassify.best_params_
gridclassify.cv_results_
"""
#by running above part of the program we find that the model which is performing the best is logistic regression for the tweets with the current parameters
#running both with best parameter found with GridSearch algorithm
"""
rfgrid = Pipeline([('ngramRF',TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)),('classifierRF',RandomForestClassifier(n_estimators=300,n_jobs=3,max_depth=12))])
rfgrid.fit(train_tweets['text'],train_tweets['class_type'])
predictrf = rfgrid.predict(test_tweets['text'])
results(rfgrid)

logrgrid = Pipeline([('ngramLOGR',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)),('classifierLOGR',LogisticRegression(penalty="l2")])
logrgrid.fit(train_tweets['text'],train_tweets['class_type'])
logrpredict = logrgrid.predict(test_tweets['text'])
results(logrgrid)
"""

"""
by running both random forest and logistic regression with GridSearch's best parameter estimation, we found that for logistic regression with
bag of words has better accuracy with the parameter estimated. Hence, that is the final model we choose in the end for our classification. 
"""

#save the pickle file
pickle.dump(pLogR,open('logR.pickle','wb'))