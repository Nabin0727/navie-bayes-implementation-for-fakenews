#!/usr/bin/env python
# coding: utf-8

# In[63]:


# Including Libraries
# necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import itertools
import numpy as np
import seaborn as sb
import pickle
from nltk.corpus import wordnet


# In[64]:


#We will import the file through the pandas 
TrueNews = pd.read_csv("True.csv")
FakeNews = pd.read_csv("Fake.csv")
#Let's check our data
print(TrueNews.head(5))
#print(FakeNews.head(5))


# In[65]:


#Checking shape for both files
print(TrueNews.shape)
print(FakeNews.shape)


# In[66]:


print('FAKE',FakeNews.isnull().sum())
print('TRUE',TrueNews.isnull().sum())


# In[67]:


#Columns Print
print(list(TrueNews.columns))
print(list(FakeNews.columns))


# In[68]:


#We are adding label fake and true
TrueNews['label'] = 'True'
TrueNews.head(5)


# In[69]:


FakeNews['label'] = 'Fake'
FakeNews.head(5)


# In[70]:


TrueNews.head()


# In[71]:


#Let's concatenate the dataframes
frames = [TrueNews, FakeNews]
news_dataset = pd.concat(frames)
news_dataset


# In[ ]:





# In[72]:


#New combined dataset 
news_dataset.describe()





news_dataset.info()





final_data = news_dataset.dropna()





final_data.isnull().sum()





# Removing the date 
final_data.drop(["date"],axis=1,inplace=True)
final_data.head()


# In[77]:


# Removing the title
final_data.drop(["title"],axis=1,inplace=True)
final_data.head()


# In[78]:


#First lets convert our data into lower case 
final_data['text'] = final_data['text'].apply(lambda x: x.lower())
final_data.head()


# In[79]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[80]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[81]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[82]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[83]:


#New combined dataset 
final_data.describe()


# In[84]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[85]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[86]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[87]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[88]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[89]:


# Drop odd rows
final_data = final_data.iloc[::2]


# In[90]:


#New combined dataset 
final_data.describe()


# In[91]:


#Removing punctuation
import string

def remove_punctuation(text):
    #all_list = [char for char in text if char not in string.punctuation]
    #no_punct = ''.join(all_list)
    translator = str.maketrans('', '',string.punctuation)
    no_punct = text.translate(translator)
    return no_punct

final_data['text'] = final_data['text'].apply(remove_punctuation)


# In[92]:


# Verifying
final_data.head()


# In[93]:


# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

final_data['text'] = final_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[94]:


final_data.head()


# In[95]:


final_data = final_data.sample(frac = 1)


# In[96]:


final_data.head()


# In[97]:


final_data.describe()


# In[98]:


final_data.to_csv('final_data.csv')


# In[99]:


# Function to plot the confusion matrix 
# This function prints and plots the confusion matrix
# Normalization can be applied by setting 'normalize=True'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[100]:


x = final_data['text']
y = final_data['label']


# In[101]:


# Convert the class labels from strings to integers
y = np.array([1 if label == "True" else 0 for label in y])


# In[102]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[103]:


# Transform the training data into bag of words features using the CountVectorizer
count_vectorizer = CountVectorizer()

x_train_bow = count_vectorizer.fit_transform(x_train)


# In[104]:


# Transform the test data into bag of words features using the CountVectorizer

x_test_bow = count_vectorizer.transform(x_test)


# In[105]:


# Get the feature names of 'count_vectorizer'

print(count_vectorizer.get_feature_names()[:10])


# In[106]:


def train_naive_bayes(X, y):
    # Calculate the number of fake and real news articles in the training set
    y = y.astype(int)
    num_fake = np.sum(y)
    num_real = len(y) - num_fake
    
    # Calculate the probability of a news article being fake or real
    p_fake = num_fake / len(y)
    p_real = num_real / len(y)
    
    # Calculate the probability of each word appearing in a fake or real news article
    fake_word_probs = {}
    real_word_probs = {}
    for i in range(X.shape[0]):
        article = X[i, :]
        nonzero_indices = article.nonzero()[1]
        for index in nonzero_indices:
            word = count_vectorizer.get_feature_names()[index]
            if y[i] == 1:  # News article is fake
                if word in fake_word_probs:
                    fake_word_probs[word] += 1
                else:
                    fake_word_probs[word] = 1
            else:  # News article is real
                if word in real_word_probs:
                    real_word_probs[word] += 1
                else:
                    real_word_probs[word] = 1
    
    # Normalize the word counts to obtain probabilities
    for word in fake_word_probs:
        fake_word_probs[word] /= num_fake
    for word in real_word_probs:
        real_word_probs[word] /= num_real
    
    return p_fake, p_real, fake_word_probs, real_word_probs


# In[107]:


def predict_fake_news(X, p_fake, p_real, fake_word_probs, real_word_probs):
    y_pred = []
    for i in range(X.shape[0]):
        article = X[i, :]
        p_real_article = 1.0
        p_fake_article = 1.0
        nonzero_indices = article.nonzero()[1]
        for index in nonzero_indices:
            word = count_vectorizer.get_feature_names()[index]
            if word in fake_word_probs:
                p_fake_article *= fake_word_probs[word]
            if word in real_word_probs:
                p_real_article *= real_word_probs[word]
        
        # Predict the class label of the news article
        if p_fake_article > p_real_article:
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    return np.array(y_pred)


# In[ ]:





# In[111]:


# Train the Naive Bayes classifier
p_fake, p_real, fake_word_probs, real_word_probs = train_naive_bayes(x_train_bow, y_train)


# Predict the class labels of the test set
y_pred = predict_fake_news(x_test_bow, p_fake, p_real, fake_word_probs, real_word_probs)


# In[112]:


# Evaluate the performance of the classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:





# In[ ]:




