# Importing Notebooks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing constants
TOKEN_SPAM_PROB_FILE = 'SpamData/03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = 'SpamData/03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = 'SpamData/03_Testing/prob-all-tokens.txt'

TEST_FEATURE_MATRIX = 'SpamData/03_Testing/test-features.txt'
TEST_TARGET_FILE = 'SpamData/03_Testing/test-target.txt'

VOCAB_SIZE = 2500

# Load the data

# Faetures
X_test = np.loadtxt(TEST_FEATURE_MATRIX, delimiter= ' ')
# Target
y_test = np.loadtxt(TEST_TARGET_FILE, delimiter = ' ')
# Token Probabilites
prob_token_spam = np.loadtxt(TOKEN_SPAM_PROB_FILE, delimiter = ' ')
prob_token_ham = np.loadtxt(TOKEN_HAM_PROB_FILE, delimiter = ' ')
prob_all_tokens = np.loadtxt(TOKEN_ALL_PROB_FILE, delimiter = ' ')

# Calculating Joint Probability

# Set the prior
PROB_SPAM = 0.3116
np.log(prob_token_spam)

# Joint probability in log format for spam
# P(SPAM | X) = P(X| SPAM)*P(SPAM) / P(X)
joint_log_spam = X_test.dot(np.log(prob_token_spam) - np.log(prob_all_tokens)) + np.log(PROB_SPAM)

# Joint probability in log format for non-spam
# # P(HAM | X) = P(X| HAM)*P(1 -SPAM) / P(X)
joint_log_ham = X_test.dot(np.log(prob_token_ham) - np.log(prob_all_tokens)) + np.log(1- PROB_SPAM)

# Making Prediction
prediction = joint_log_spam > joint_log_ham

# Simplify
joint_log_spam = X_test.dot(np.log(prob_token_spam)) + np.log(PROB_SPAM)
joint_log_ham = X_test.dot(np.log(prob_token_ham)) + np.log(1-PROB_SPAM)

# Metrics and Evaluation
# Accuracy
correct_docs = (y_test == prediction).sum()
print('Docs classified correctly', correct_docs)
numdocs_wrong = X_test.shape[0] - correct_docs
print('Docs classified incorrectly', numdocs_wrong)
# Accuracy
accuracy = correct_docs/len(X_test)
print('Accuracy of the model is {:.2%}'.format(accuracy))

# Visualizing the results
# Chart Styling Info
yaxis_label = 'P(X | Spam)'
xaxis_label = 'P(X | Nonspam)'

linedata = np.linspace(start=-14000, stop=1, num=1000)
# Chart Nr 1:
plt.subplot(1, 2, 1)

plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)

# Set scale
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=25)
plt.plot(linedata, linedata, color='orange')

# Chart Nr 2:
plt.subplot(1, 2, 2)

plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)

# Set scale
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=3)
plt.plot(linedata, linedata, color='orange')

plt.show()

# False Positive and False Negative
np.unique(prediction, return_counts = True)
# True Positive
true_pos = (y_test == 1) & (prediction == 1)
true_pos.sum()
# False Positive
false_pos = (y_test == 0) & (prediction ==1)
false_pos.sum()
# False Negative
false_neg = (y_test == 1) & (prediction == 0)
false_neg.sum()

# Recall Score
recall_score = true_pos.sum() / (true_pos.sum() + false_neg.sum())
print('Recall Score is {:.2%}'.format(recall_score))

# Precision Score
precision_score = true_pos.sum() / (true_pos.sum() + false_pos.sum())
print('Precision Score is {:.3}'.format(precision_score))

# F-Score or F1 Score
f_score = 2* recall_score * precision_score/(recall_score + precision_score)
print('F-Score os {:.2}'.format(f_score))









