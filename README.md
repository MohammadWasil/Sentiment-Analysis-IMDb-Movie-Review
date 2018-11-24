# IMBD-Movie-Review

![image](https://user-images.githubusercontent.com/31696557/48961509-28e70c00-ef9b-11e8-85f3-86a584d823de.png)

Sentiment Analysis on IMBD Movie Reviews.

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

### Sentiment Analysis Using Bags of Words - Random Forest

  1) Change the directory, in read_csv(), to location of your labeledTrainData.tsv.<br/>
  2) Change the directory, in read_csv(), to location of your testData.tsv.
  3) Run the file.
  
### Sentiment Analysis Using RNN - Recurrent Neural Network.
  
  1) Change the directory of data_train and data_test of ''Sentiment Analysis using RNN' to the location of respective dataset.<br/>
  2) Run the file.<br/>
  3) First, it will ask for the input for methods of preprocessing the data - which are - Porter Stemming, Stop Wrods, or Neither of them. Accordingly, it will process the data.<br/>
  4) Then, it will ask for the input for model - LSTM RNN or GRU RNN.
  5) Compile the model, and it will create a csv file for the predicted sentiment of test data.
  
  6) Now, to predict your own review, run 'Predict Class For IMBD Movie Review.py'.
  7) It will ask for which model to use, which methods of preprocessing to use, and then it will predict the sentiment of the review.
  
 To know more about Recurrent Neural Network, check [this](https://www.coursera.org/learn/nlp-sequence-models) course.


