# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:45:13 2018

@author: Mohammad Wasil Saleem
"""
# Sentimnetal Analysis Using Bags Of Words.

import pandas as pd
import re
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def reviewWords(review):
    data_train_Exclude_tags = re.sub(r'<[^<>]+>', " ", review)      # Excluding the html tags
    data_train_num = re.sub(r'[0-9]+', 'number', data_train_Exclude_tags)  # Converting numbers to "NUMBER"
    data_train_lower = data_train_num.lower()              # Converting to lower case.
    data_train_split = data_train_lower.split()            # Splitting into individual words.
    stopWords = set(stopwords.words("english") )

    meaningful_words = [w for w in data_train_split if not w in stopWords]     # Removing stop words.
    
    return( " ".join( meaningful_words ))   

# Reading the Data
data_train = pd.read_csv('.../labeledTrainData.tsv',delimiter = "\t")

# Data Cleaing And Text Prepocessing.

'''
We need to decide how to deal with frequently occurring words that don't carry much meaning. 
Such words are called "stop words"; in English they include words such as "a", "and", "is", and "the". 
'''

# To get a list of stop words:
print("List of stop words!")
print(stopwords.words("english") )
print("---Ended---\n")

# Let's process all the reviews together.
cleanWords = []
for i in range(data_train['review'].size):
    #print('Processin', i)
    cleanWords.append( reviewWords( data_train["review"][i] ))
print("---Review Processing Done!---\n")

# Creating features from bags of words.
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
data_train_features = vectorizer.fit_transform(cleanWords)
#data_train_features = data_train_features.toarray()         # 25000x5000 sparse matrix, with 2105457 stored elements in compressed Sparse Row format.
print("Features Created!!!\n")

# Training
print("Training the classifier\n")
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(data_train_features, data_train["sentiment"])
score = forest.score(data_train_features, data_train["sentiment"])
print("Mean Accuracy of the Random forest is: %f" %(score))

# Predicting the model.

data_test = pd.read_csv('.../testData.tsv', delimiter = "\t")

# Let's process all the test reviews together.
testcleanWords = []
for i in range(data_test['review'].size):
    #print('Processin', i)
    testcleanWords.append( reviewWords( data_test["review"][i] ))
print("---Review Processing Done!---\n")

# Creating features from bags of words.
data_test_features = vectorizer.transform(testcleanWords)
#data_train_features = data_train_features.toarray()         # 25000x5000 sparse matrix, with 2105457 stored elements in compressed Sparse Row format.
print("Test Features Created!!!\n")

# Making Predictions.
result = forest.predict(data_test_features)

# Copy the predicted values to pandas dataframe with an id column, and a sentiment column.
output = pd.DataFrame(data = {"id": data_test["id"], "sentiment": result} )

output.to_csv("predictedResult.csv", index = False, quoting = 3 )
# Score on kaggle comes out to be 0.84176
