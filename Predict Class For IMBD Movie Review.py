# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 19:50:54 2018

@author: solutions
"""
import re
import pickle
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem import PorterStemmer

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def reviewWords(review, method):
    data_train_Exclude_tags = re.sub(r'<[^<>]+>', " ", review)      # Excluding the html tags
    data_train_num = re.sub(r'[0-9]+', 'number', data_train_Exclude_tags)  # Converting numbers to "NUMBER"
    data_train_lower = data_train_num.lower()              # Converting to lower case.
    data_train_no_punctuation = re.sub(r"[^a-zA-Z]", " ", data_train_lower )
       
    # using porter stemming.
    # https://pythonprogramming.net/stemming-nltk-tutorial/
    # https://github.com/MohammadWasil/Coursera-Machine-Learning-Python/blob/master/CSR%20ML/WEEK%237/Machine%20Learning%20Assignment%236/Python/processEmail.py
    if method == "Porter Stemming":
        #print("Processing dataset with porter stemming...")
        stemmedWords = [ps.stem(word) for word in re.findall(r"\w+", data_train_no_punctuation)]
        return(" ".join(stemmedWords))         
        
    # using stop words.
    # After using stop words, training accuracy increases, but testing accuracy decreases in Kaggle.
    # This method might overfit the training data.
    if method == "Stop Words":
        #print("Processing dataset with stop words...")
        data_train_split = data_train_no_punctuation.split()            # Splitting into individual words.
        stopWords = set(stopwords.words("english") )
        meaningful_words = [w for w in data_train_split if not w in stopWords]     # Removing stop words.
        return( " ".join( meaningful_words ))  
    
    if method == "Nothing":
        #print("Processing dataset without porter stemming and stop words...")
        return data_train_no_punctuation 



# loading the tokenizer, which we have saved during the training step.
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 500           # Input for keras.

# Load the model.
# Loading the one model out of 6 saved models, according to user's choice.
print("Enter 'LSTM' for RNN LSTM model, or 'GRU' for RNN GRU model.\n")
modelSelect = input("Which model do you want to use?\n")

print("Input 'Porter Stemming' for porter stemming, 'Stop Words' for stop words, or anywords for Neither of them: ")
if modelSelect == "LSTM":
    modelSelected = 'LSTM'
    preprocessingInput = input("Do you want to include porter stemming, Stop Words or None of them?")
            
    if preprocessingInput == "Porter Stemming":
        method = "Porter Stemming"
        ps = PorterStemmer()        # instantiating a class instance.
        generateNameForModel = "RNN " + modelSelected + " model" + method + ".h5"        
        # Loading the saved model.
        themodel = load_model(generateNameForModel)

        
    elif preprocessingInput == "Stop Words":
        method = "Stop Words"
        
        generateNameForModel = "RNN " + modelSelected + " model" + method + ".h5"        
        # Loading the saved model.
        themodel = load_model(generateNameForModel)
        
    else:
        method = "Nothing"
        
        generateNameForModel = "RNN " + modelSelected + " model" + method + ".h5"        
        # Loading the saved model.
        themodel = load_model(generateNameForModel)
    
else:
    modelSelected = 'GRU'
    preprocessingInput = input("Do you want to include porter stemming, Stop Words or None of them?")
            
    if preprocessingInput == "Porter Stemming":
        method = "Porter Stemming"
        ps = PorterStemmer()        # instantiating a class instance.
        generateNameForModel = "RNN " + modelSelected + " model" + method + ".h5"        
        # Loading the saved model.
        themodel = load_model(generateNameForModel)

        
    elif preprocessingInput == "Stop Words":
        method = "Stop Words"
        
        generateNameForModel = "RNN " + modelSelected + " model" + method + ".h5"        
        # Loading the saved model.
        themodel = load_model(generateNameForModel)
        
    else:
        method = "Nothing"
        
        generateNameForModel = "RNN " + modelSelected + " model" + method + ".h5"        
        # Loading the saved model.
        themodel = load_model(generateNameForModel)
        
        
        
# Input the text and predict its value.
for i in range(1000):
    inputReview = input("Enter the review to get the sentiment: ")
    
    if inputReview == "Quit" or inputReview == "quit":
        break
    else:
        # Processing text set reviews.
        testcleanWords = []
        
        testcleanWords.append( reviewWords( inputReview ), method)
        print("---Test Review Processing Done!---\n")
        
        #tokenising Test data
        test_review_tokenized = tokenizer.texts_to_sequences(testcleanWords)
        x_test = pad_sequences(test_review_tokenized, maxlen = MAX_REVIEW_LENGTH_FOR_KERAS_RNN)       
        
        # Prediction.
        ytest_prediction = themodel.predict(x_test)
        ytest_prediction = round(ytest_prediction)
        ytest_prediction = ytest_prediction.astype(int)
        
        print("The sentiment is: ", ytest_prediction)