# Using Sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score, precision_score, f1_score

DATA_JSON_FILE = 'SpamData/01_Processing/email-text-data.json'

data = pd.read_json(DATA_JSON_FILE)
data.sort_index(inplace = True)

# Creating vocabulary
vectorizer = CountVectorizer(stop_words = 'english')
all_features = vectorizer.fit_transform(data.MESSAGE)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(all_features, data.CATEGORY, test_size = 0.3, random_state=88 )

# Training the model
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting from trained model
prediction = classifier.predict(X_test)

nr_correct = (y_test == prediction).sum()
nr_incorrect = y_test.size - nr_correct

# Accuracy & Metrics Calculations
accuracy = nr_correct/(nr_correct + nr_incorrect)
recall = recall_score(y_test, prediction)
precision = precision_score(y_test, prediction)
f_score= f1_score(y_test, prediction)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

# Testing our model on a sample list
example = ['Hey\' How are you man', 
           'Nice to meet you Sharry',
           'I want to pee',
           'Today was a nice day to play cricket', 
           'get a viagra for free now',
           'Kidda bai ki haal chal']
transformed = vectorizer.transform(example)
classifier.predict(transformed)








