import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

import re
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

from datetime import datetime
from translate import Translator

translator= Translator(to_lang="ru")
index = 1

train1 = pd.read_json("train1.json", encoding = 'utf-8') # imdb train (50000)
train2 = pd.read_json("train2.json", encoding = 'utf-8') # hackaton train (8000)
train3 = pd.read_json("train3.json", encoding = 'utf-8') # answers of train1 & train2
# train3 = pd.read_json("untitled.json", encoding = 'utf-8') # answers of train2 
# train3 = pd.read_json("untitled2.json", encoding = 'utf-8') # answers of train1

print ("Read train json!")

def review_to_words( raw_review, eng ):
  global index
  # Function to convert a raw review to a string of words
  # The input is a single string (a raw movie review), and
  # the output is a single string (a preprocessed movie review)
  #
  # 1. Remove HTML
  review_text = BeautifulSoup(raw_review, 'lxml').get_text()
  #
  # 2. Remove non-letters        
  if eng:
    letters_only = re.sub("[^a-zA-Z^]", " ", review_text)
  else:
    letters_only = re.sub("[^а-яА-Я^]", " ", review_text)

  #
  # 3. Convert to lower case, split into individual words

  if eng:
    letters_only = translator.translate(letters_only)

    print(index)
    index += 1

    words = letters_only.lower().split()
  else:
    words = letters_only.lower().split()

  #
  # 4. In Python, searching a set is much faster than searching
  #   a list, so convert the stop words to a set

  stops = set(stopwords.words("russian"))
  #
  # 5. Remove stop words
  
  meaningful_words = [w for w in words if not w in stops]
  #
  # 6. Join the words back into one string separated by space,
  # and return the result.
  return( " ".join( meaningful_words ))

print ("Function review_to_words declared!")


clean_train_texts = []

  #Cleaning text from 
num_texts1 = train1["text"].size
for i in range( 0, num_texts1 ):                    
  clean_train_texts.append( review_to_words( train1["text"][i], True ))

print ("Added train1 to clean_train_texts")

num_texts2 = train2["text"].size
for i in range( 0, num_texts2 ):                    
  clean_train_texts.append( review_to_words( train2["text"][i], False ))

print ("Added train2 to clean_train_texts")

#Adding features to each text
vectorizer = CountVectorizer(analyzer = "word",   \
                           tokenizer = None,    \
                           preprocessor = None, \
                           stop_words = None,   \
                           max_features = 5000) 
print ("Created vectorizer")

train_data_features = vectorizer.fit_transform(clean_train_texts)
train_data_features = train_data_features.toarray()
#creating dictionary with 5000 words
vocab = vectorizer.get_feature_names()
#counting frequency of each word
print ("Training the random forest...")

start_time = datetime.now()

forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit( train_data_features, train3["sentiment"] )

end_time = datetime.now()

print ("Training completed!")

difference = end_time - start_time
print(difference.total_seconds())
print("seconds")

# Read the test data
test = pd.read_json("test.json", encoding = 'utf-8')

print ("Read test json")

# Create an empty list and append the clean reviews one by one
num_texts = len(test["text"])
clean_test_texts = []


print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_texts):
   clean_text = review_to_words( test["text"][i], False )
   clean_test_texts.append( clean_text )

print ("Added test json to clean_test_texts!")

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_texts)
test_data_features = test_data_features.toarray()
# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

print ("Prediction ended!")

output = pd.DataFrame( data={"sentiment":result} )
output.to_json( "empty.json")
predict_data1 = pd.read_json("empty.json", encoding = 'utf-8')

print ("Read empty json")

answer_data1 = pd.read_json("capitals.json", encoding = 'utf-8')

print ("Read capitals json")

num_iter = test.shape[0]
counter = 0
for i in range(0, num_iter):
   if (predict_data1["sentiment"][i]==answer_data1["sentiment"][i]):
       counter+=1
   else:
       counter = counter
accuracy = (counter/num_iter)*100
print (accuracy)