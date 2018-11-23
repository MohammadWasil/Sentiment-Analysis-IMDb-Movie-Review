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

LabeledTrainData has 25000 rows containing 3 columns - id, Sentiment, review.
TestData has 25000 rows containing only 2 columns - id, and reviews. We have to predict the sentiments of these reviews.






