# IMBD-Movie-Review

IMBD - IMDb (Internet Movie Database) is an online database of information related to films, television programs, home videos and video games, and internet streams, including cast, production crew and personnel biographies, plot summaries, trivia, and fan reviews and ratings.




## Dependencies

You can install dependencies by running the following command in Anaocnda prompt:

```
# Theano
conda install mingw libpython
conda install mkl=2017.0.3

# Keras
pip install keras
```
We also need NLTK(Natural Language ToolKit) package. It is already installed in Anaconda.

If it is not installed, you can install it by running the commands in Anaconda prompt:
```
# NLTK
conda install -c conda-forge nltk 
```

After downloading NLTK package, we need to download NLTK dataset.

```
import nltk
nltk.download()
```
This window will pop-up. 

![nltk downloader](https://user-images.githubusercontent.com/31696557/48960208-bec96980-ef90-11e8-8b46-ad09eb675461.png)

Download All Packages.

## Dataset

Two files - 1) labeledTrainData 2) testData.
These datasets have been downloaded from Kaggle Competition - Bags of Words Meets Bags Of Popcorn.

LabeledTrainData has 25000 rows containing 3 columns - id, Sentiment, review.<br/>
TestData has 25000 rows containing only 2 columns - id, and reviews. We have to predict the sentiments of these reviews.

## Sentiment Analysis

Sentiment Analysis of IMBD Movie datasets is done using two different machine learning algorithm:
1) Random forest
2) Recurrent Neural Network.

First, we trained the model using Random Forest. It has a training accuracy of   , and score on kaggle comes out to be 0.84176.

We also trained the model on LSTM and GRU Recurrent Neural Network, using different preprocessing techniques, like Porter stemming, Stop words etc. It gives training accuracy in range of 91.57 to 92.76, and score on Kaggle comes in the range of 0.86768 to 0.87896.

## How to work with the code






