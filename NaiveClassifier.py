# Importing Libraries
import pandas as pd
import numpy as np

# Importing constants
TRAINING_DATA_FILE = 'SpamData/02_Training/train-data.txt'
TEST_DATA_FILE = 'SpamData/02_Training/test-data.txt'

TOKEN_SPAM_PROB_FILE = 'SpamData/03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = 'SpamData/03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = 'SpamData/03_Testing/prob-all-tokens.txt'

TEST_FEATURE_MATRIX = 'SpamData/03_Testing/test-features.txt'
TEST_TARGET_FILE = 'SpamData/03_Testing/test-target.txt'

VOCAB_SIZE = 2500

# Read and Load data from .txt files into Numpy Array
sparse_train_data = np.loadtxt(TRAINING_DATA_FILE, delimiter=' ', dtype=int)
sparse_test_data = np.loadtxt(TEST_DATA_FILE, delimiter = ' ', dtype =int)

# Create Empty DataFrame
column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, VOCAB_SIZE))
column_names[:5]
index_names = np.unique(sparse_train_data[:, 0])
index_names
full_train_data = pd.DataFrame(index=index_names, columns=column_names)
full_train_data.fillna(value=0, inplace=True)
full_train_data.head()

# Create a full matrix from Sparse Matrix
def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1, cat_idx=2, freq_idx=3):
    """
    Form a full matrix from a sparse matrix. Return a pandas dataframe. 
    Keyword arguments:
    sparse_matrix -- numpy array
    nr_words -- size of the vocabulary. Total number of tokens. 
    doc_idx -- position of the document id in the sparse matrix. Default: 1st column
    word_idx -- position of the word id in the sparse matrix. Default: 2nd column
    cat_idx -- position of the label (spam is 1, nonspam is 0). Default: 3rd column
    freq_idx -- position of occurrence of word in sparse matrix. Default: 4th column
    """
    column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, VOCAB_SIZE))
    doc_id_names = np.unique(sparse_matrix[:, 0])
    full_matrix = pd.DataFrame(index=doc_id_names, columns=column_names)
    full_matrix.fillna(value=0, inplace=True)
    
    for i in range(sparse_matrix.shape[0]):
        doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        label = sparse_matrix[i][cat_idx]
        occurrence = sparse_matrix[i][freq_idx]
        
        full_matrix.at[doc_nr, 'DOC_ID'] = doc_nr
        full_matrix.at[doc_nr, 'CATEGORY'] = label
        full_matrix.at[doc_nr, word_id] = occurrence
    
    full_matrix.set_index('DOC_ID', inplace=True)
    return full_matrix

full_train_data = make_full_matrix(sparse_train_data, VOCAB_SIZE)

# ---- Training Naive Bayes Model ----

# Calculating prob. of spam
full_train_data.CATEGORY.size
full_train_data.CATEGORY.sum()
prob_spam = full_train_data.CATEGORY.sum() / full_train_data.CATEGORY.size
print('Probability of spam is', prob_spam)

# Total no. of words
full_train_features = full_train_data.loc[:, full_train_data.columns != 'CATEGORY']
full_train_features.head()
email_lengths = full_train_features.sum(axis=1)
email_lengths.shape

total_wc = email_lengths.sum()
total_wc

spam_lengths = email_lengths[full_train_data.CATEGORY == 1]
spam_lengths.shape
spam_wc = spam_lengths.sum()
spam_wc

ham_lengths = email_lengths[full_train_data.CATEGORY == 0]
ham_lengths.shape
nonspam_wc = ham_lengths.sum()
nonspam_wc

# Summing the tokens in spam
train_spam_tokens = full_train_features.loc[full_train_data.CATEGORY == 1]
train_spam_tokens.head()

summed_spam_tokens = train_spam_tokens.sum(axis=0) + 1
summed_spam_tokens.shape
summed_spam_tokens.tail()

# Summing the tokens in ham
train_ham_tokens = full_train_features.loc[full_train_data.CATEGORY == 0]
summed_ham_tokens = train_ham_tokens.sum(axis=0) + 1
# P(TOKEN|SPAM) 
prob_tokens_spam = summed_spam_tokens / (spam_wc + VOCAB_SIZE)
prob_tokens_spam[:5]
prob_tokens_spam.sum()

# P(TOKEN|HAM) 
prob_tokens_nonspam = summed_ham_tokens / (nonspam_wc + VOCAB_SIZE)
prob_tokens_nonspam.sum()


# P(TOKEN)
prob_tokens_all = full_train_features.sum(axis=0) / total_wc
prob_tokens_all.sum()

# Saving the train model
np.savetxt(TOKEN_SPAM_PROB_FILE, prob_tokens_spam)
np.savetxt(TOKEN_HAM_PROB_FILE, prob_tokens_nonspam)
np.savetxt(TOKEN_ALL_PROB_FILE, prob_tokens_all)

# Preparing test data
full_test_data = make_full_matrix(sparse_test_data, nr_words=VOCAB_SIZE)
X_test = full_test_data.loc[:, full_test_data.columns != 'CATEGORY']
y_test = full_test_data.CATEGORY

np.savetxt(TEST_TARGET_FILE, y_test)
np.savetxt(TEST_FEATURE_MATRIX, X_test)








