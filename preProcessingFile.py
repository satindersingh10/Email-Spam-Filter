# Importing notebooks
from os import walk
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
from wordcloud import WordCloud
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split

# %matplotlib inline

EXAMPLE_FILE = 'SpamData/01_Processing/practice_email.txt'

SPAM_1_PATH = 'SpamData/01_Processing/spam_assassin_corpus/spam_1'
SPAM_2_PATH = 'SpamData/01_Processing/spam_assassin_corpus/spam_2'
EASY_NONSPAM_1_PATH = 'SpamData/01_Processing/spam_assassin_corpus/easy_ham_1'
EASY_NONSPAM_2_PATH = 'SpamData/01_Processing/spam_assassin_corpus/easy_ham_2'

SPAM_CAT = 1
HAM_CAT = 0
VOCAB_SIZE = 2500

DATA_JSON_FILE = 'SpamData/01_Processing/email-text-data.json'
WORD_ID_FILE = 'SpamData/01_Processing/word-by-id.csv'

TRAINING_DATA_FILE = 'SpamData/02_Training/train-data.txt'
TEST_DATA_FILE = 'SpamData/02_Training/test-data.txt'

WHALE_FILE = 'SpamData/01_Processing/wordcloud_resources/whale-icon.png'
SKULL_FILE = 'SpamData/01_Processing/wordcloud_resources/skull-icon.png'
THUMBS_UP_FILE = 'SpamData/01_Processing/wordcloud_resources/thumbs-up.png'
THUMBS_DOWN_FILE = 'SpamData/01_Processing/wordcloud_resources/thumbs-down.png'
CUSTOM_FONT_FILE = 'SpamData/01_Processing/wordcloud_resources/OpenSansCondensed-Bold.ttf'

# Example Email
stream = open(EXAMPLE_FILE, encoding='latin-1')
message = stream.read()
stream.close()
print(type(message))
print(message)
import sys
sys.getfilesystemencoding()
stream = open(EXAMPLE_FILE, encoding='latin-1')
is_body = False
lines = []

for line in stream:
    if is_body:
        lines.append(line)
    elif line == '\n':
        is_body = True

stream.close()

email_body = '\n'.join(lines)
print(email_body)

#  --- Email body generator ---
def email_body_generator(path):
    
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            
            filepath = join(root, file_name)
            
            stream = open(filepath, encoding='latin-1')

            is_body = False
            lines = []

            for line in stream:
                if is_body:
                    lines.append(line)
                elif line == '\n':
                    is_body = True

            stream.close()

            email_body = '\n'.join(lines)
            
            yield file_name, email_body
            

def df_from_directory(path, classification):
    rows = []
    row_names = []
    
    for file_name, email_body in email_body_generator(path):
        rows.append({'MESSAGE': email_body, 'CATEGORY': classification})
        row_names.append(file_name)
        
    return pd.DataFrame(rows, index=row_names)

# Creating DataFrame for spam emails
spam_emails = df_from_directory(SPAM_1_PATH, 1)
spam_emails = spam_emails.append(df_from_directory(SPAM_2_PATH, 1))
spam_emails.head()

# Creating DataFrame for non-spam emails
ham_emails = df_from_directory(EASY_NONSPAM_1_PATH, HAM_CAT)
ham_emails = ham_emails.append(df_from_directory(EASY_NONSPAM_2_PATH, HAM_CAT))

# Concatinating two dataframes
data = pd.concat([spam_emails, ham_emails])

# Data Cleaning: Checking for Missing Values

# check if any message bodies are null
data['MESSAGE'].isnull().values.any()

# check if there are empty emails (string length zero)
(data.MESSAGE.str.len() == 0).any()

# Challenge: how would you check the number of entries with null/None values?
data.MESSAGE.isnull().sum()

# Locate empty emails
data[data.MESSAGE.str.len() == 0].index
data.index.get_loc('.DS_Store')
data[4608:4611]

# Remove System File Entries from Dataframe

data.drop(['cmds', '.DS_Store'], inplace=True)
data[4608:4611]

# Add Document IDs to Track Emails in Dataset
document_ids = range(0, len(data.index))
data['DOC_ID'] = document_ids
data['FILE_NAME'] = data.index
data.set_index('DOC_ID', inplace=True)
data.head()

# Save to File using Pandas
data.to_json(DATA_JSON_FILE)

# --- Number of Spam Messages Visualised (Pie Charts) ---
data.CATEGORY.value_counts()
amount_of_spam = data.CATEGORY.value_counts()[1]
amount_of_ham = data.CATEGORY.value_counts()[0]

category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90, 
       autopct='%1.0f%%')
plt.show()

category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]
custom_colours = ['#ff7675', '#74b9ff']

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90, 
       autopct='%1.0f%%', colors=custom_colours, explode=[0, 0.1])
plt.show()

# --- Natural Language Processing ---
# Text Pre-Processing

# convert to lower case
msg = 'All work and no play makes Jack a dull boy.'
msg.lower()

# Download the NLTK Resources (Tokenizer & Stopwords)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('shakespeare')

# Tokenising
msg = 'All work and no play makes Jack a dull boy.'
word_tokenize(msg.lower())

# Removing Stop Words
stop_words = set(stopwords.words('english'))

# Word Stems and Stemming
# Removing HTML tags from Emails
soup = BeautifulSoup(data.at[2, 'MESSAGE'], 'html.parser')
soup.get_text()

# Function for Email Processing
def clean_message(message, stemmer=PorterStemmer(), 
                 stop_words=set(stopwords.words('english'))):
    
    # Converts to Lower Case and splits up the words
    words = word_tokenize(message.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
    
    return filtered_words
clean_message(email_body)

# Challenge: Modify function to remove HTML tags. Then test on Email with DOC_ID 2. 
def clean_msg_no_html(message, stemmer=PorterStemmer(), 
                 stop_words=set(stopwords.words('english'))):
    
    # Remove HTML tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()
    
    # Converts to Lower Case and splits up the words
    words = word_tokenize(cleaned_text.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
#             filtered_words.append(word) 
    
    return filtered_words

clean_msg_no_html(data.at[2, 'MESSAGE'])

# --- Apply Cleaning and Tokenisation to all messages

# Slicing Dataframes and Series & Creating Subsets
data.iat[2, 2]
data.iloc[5:11]
first_emails = data.MESSAGE.iloc[0:3]

nested_list = first_emails.apply(clean_message)
# flat_list = []
# for sublist in nested_list:
#     for item in sublist:
#         flat_list.append(item)

flat_list = [item for sublist in nested_list for item in sublist]
        
len(flat_list)

# use apply() on all the messages in the dataframe
nested_list = data.MESSAGE.apply(clean_msg_no_html)

# Using Logic to Slice Dataframes
data[data.CATEGORY == 1].shape
data[data.CATEGORY == 1].tail()
doc_ids_spam = data[data.CATEGORY == 1].index
doc_ids_ham = data[data.CATEGORY == 0].index

#Subsetting a Series with an Index
nested_list_ham = nested_list.loc[doc_ids_ham]
nested_list_ham.shape
nested_list_ham.tail()
nested_list_spam = nested_list.loc[doc_ids_spam]

#  use python list comprehension and then find the total number of 
# words in our cleaned dataset of spam email bodies. Also find the total number of 
# words in normal emails in the dataset. Then find the 10 most common words used in 
# spam. Also, find the 10 most common words used in non-spam messages. 

flat_list_ham = [item for sublist in nested_list_ham for item in sublist]
normal_words = pd.Series(flat_list_ham).value_counts()

normal_words.shape[0] # total number of unique words in the non-spam messages

normal_words[:10]

flat_list_spam = [item for sublist in nested_list_spam for item in sublist]
spammy_words = pd.Series(flat_list_spam).value_counts()

spammy_words.shape[0] # total number of unique words in the spam messages
spammy_words[:10]

# --- Generate Vocabulary & Dictionary ---
stemmed_nested_list = data.MESSAGE.apply(clean_msg_no_html)
flat_stemmed_list = [item for sublist in stemmed_nested_list for item in sublist]
unique_words = pd.Series(flat_stemmed_list).value_counts()
print('Nr of unique words', unique_words.shape[0])
unique_words.head()

# Creating subset of the series called 'frequent_words' that only contains
# the most common 2,500 words out of the total. Print out the top 10 words
frequent_words = unique_words[0:VOCAB_SIZE]
print('Most common words: \n', frequent_words[:10])

# Create Vocabulary DataFrame with a WORD_ID
word_ids = list(range(0, VOCAB_SIZE))
vocab = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=word_ids)
vocab.index.name = 'WORD_ID'
vocab.head()

# Save the Vocabulary as a CSV File
vocab.to_csv(WORD_ID_FILE, index_label=vocab.index.name, header=vocab.VOCAB_WORD.name)

# Generate Features & a Sparse Matrix
# Creating a DataFrame with one Word per Column
word_columns_df = pd.DataFrame.from_records(stemmed_nested_list.tolist())
word_columns_df.head()

# Splitting the Data into a Training and Testing Dataset
X_train, X_test, y_train, y_test = train_test_split(word_columns_df, data.CATEGORY,
                                                   test_size=0.3, random_state=42)
print('Nr of training samples', X_train.shape[0])
print('Fraction of training set', X_train.shape[0] / word_columns_df.shape[0])
X_train.index.name = X_test.index.name = 'DOC_ID'
X_train.head()
y_train.head()

# Create a Sparse Matrix for the Training Data
word_index = pd.Index(vocab.VOCAB_WORD)
type(word_index[3])
word_index.get_loc('thu')

def make_sparse_matrix(df, indexed_words, labels):
    """
    Returns sparse matrix as dataframe.
    
    df: A dataframe with words in the columns with a document id as an index (X_train or X_test)
    indexed_words: index of words ordered by word id
    labels: category as a series (y_train or y_test)
    """
    
    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []
    
    for i in range(nr_rows):
        for j in range(nr_cols):
            
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.at[doc_id]
                
                item = {'LABEL': category, 'DOC_ID': doc_id,
                       'OCCURENCE': 1, 'WORD_ID': word_id}
                
                dict_list.append(item)
    
    return pd.DataFrame(dict_list)

sparse_train_df = make_sparse_matrix(X_train, word_index, y_train)
sparse_train_df[:5]

# Combine Occurrences with the Pandas groupby() Method
train_grouped = sparse_train_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
train_grouped.head()

train_grouped = train_grouped.reset_index()
train_grouped.head()

# Save Training Data as .txt File
np.savetxt(TRAINING_DATA_FILE, train_grouped, fmt='%d')
train_grouped.columns

sparse_test_df = make_sparse_matrix(X_test, word_index, y_test)
test_grouped = sparse_test_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum().reset_index()
test_grouped.head()
np.savetxt(TEST_DATA_FILE, test_grouped, fmt='%d')



















