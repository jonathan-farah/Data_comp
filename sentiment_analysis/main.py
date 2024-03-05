# created by ilayd
# march 2 2024
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd

import numpy as np

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt

#takes in text: a string array
#prepares it for analysis
#return value: string
def preprocessText(text):
    #make all the letters lower case
    tokens = word_tokenize(text.lower())

    #filter the stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    #lemmatize (get rid of -ing, -ed etc. based on parts of speech)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    #combine the text again
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

#gives the sentiment score of every given text (string array)
#return value: float
def getSentiment(text):
    #gets polarity score values from vader
    scores = analyzer.polarity_scores(text)

    #puts the compund sentiment score into sentiment variable
    sentiment = scores['compound']

    return sentiment

#takes a text(string array) and returns the number of strings with positive sentiment
def getPositive(text):
    sum = 0
    for i in text:
        if(i>0):
            sum+=1
    return sum

#takes a text(string array) and returns the number of strings with negative sentiment
def getNegative(text):
    sum = 0
    for i in text:
        if(i<0):
            sum+=1
    return sum

##takes a text(string array) and returns the number of strings with neutral sentiment

def getNeutral(text):
    sum = 0
    for i in text:
        if(i==0):
            sum+=1
    return sum

#read the data
df = pd.read_csv('Dataset_10k.csv')
#print(df)

#apply preprocessing
df['translated_title'] = df['translated_title'].apply(preprocessText)
#print(df['translated_title'])

# "no" was not in the NEGATE list in nltk.sentiment.vader file so it is added
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.pop('no')

#apply getSentiment
df['sentiment'] = df['translated_title'].apply(getSentiment)

#create a barplot for the results
x = np.array(["Positive","Negative", "Neutral"])

y = [getPositive(df['sentiment']), getNegative(df['sentiment']), getNeutral(df['sentiment'])]

plt.bar(x,y)
plt.show()

#save it to a csv file for future analysis
df.to_csv('Data10kWithSentiments.csv')



