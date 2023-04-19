#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#We will import the file through the pandas 
TrueNews = pd.read_csv("True.csv")
FakeNews = pd.read_csv("Fake.csv")
#Let's check our data
print(TrueNews.head(5))
#print(FakeNews.head(5))


# In[3]:


#Checking shape for both files
print(TrueNews.shape)
print(FakeNews.shape)


# In[4]:


print('FAKE',FakeNews.isnull().sum())
print('TRUE',TrueNews.isnull().sum())


# In[5]:


#Columns Print
print(list(TrueNews.columns))
print(list(FakeNews.columns))


# In[6]:


#We are adding label fake and true
TrueNews['label'] = 'True'
TrueNews.head(5)


# In[7]:


FakeNews['label'] = 'Fake'
FakeNews.head(5)


# In[8]:


TrueNews.head()


# In[9]:


#Let's concatenate the dataframes
frames = [TrueNews, FakeNews]
news_dataset = pd.concat(frames)
news_dataset


# In[ ]:





# In[10]:


#New combined dataset 
news_dataset.describe()


# In[11]:


news_dataset.info()


# In[12]:


final_data = news_dataset.dropna()


# In[13]:


final_data.isnull().sum()


# In[14]:


# Removing the date 
final_data.drop(["date"],axis=1,inplace=True)
final_data.head()


# In[15]:


# Removing the title
final_data.drop(["title"],axis=1,inplace=True)
final_data.head()


# In[16]:


#First lets convert our data into lower case 
final_data['text'] = final_data['text'].apply(lambda x: x.lower())
#final_data['title'] = final_data['title'].apply(lambda x: x.lower())
final_data.head()


# In[17]:


#Removing punctuation
import string

def remove_punctuation(text):
    #all_list = [char for char in text if char not in string.punctuation]
    #no_punct = ''.join(all_list)
    translator = str.maketrans('', '',string.punctuation)
    no_punct = text.translate(translator)
    return no_punct

final_data['text'] = final_data['text'].apply(remove_punctuation)


# In[18]:


# Verifying
final_data.head()


# In[19]:


# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

final_data['text'] = final_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[20]:


final_data.head()


# In[21]:


final_data = final_data.sample(frac = 1)


# In[22]:


final_data.head()


# In[23]:


final_data.describe()


# In[24]:


#final_data.to_csv('final_data.csv')


# In[25]:


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


# In[26]:


x = final_data['text']
y = final_data['label']


# In[27]:


# Convert the class labels from strings to integers
y = np.array([1 if label == "True" else 0 for label in y])


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[29]:


# Transform the training data into bag of words features using the CountVectorizer
count_vectorizer = CountVectorizer()
x_train_bow = count_vectorizer.fit_transform(x_train)


# In[30]:


# Transform the test data into bag of words features using the CountVectorizer
x_test_bow = count_vectorizer.transform(x_test)


# In[31]:


# Get the feature names of 'count_vectorizer'

print(count_vectorizer.get_feature_names()[:10])


# In[32]:


from scipy.sparse import csr_matrix

from collections import Counter

def train_naive_bayes(X, y):
    # Calculate the number of fake and real news articles in the training set
    num_fake = np.sum(y)
    num_real = len(y) - num_fake
    
    # Calculate the probability of a news article being fake or real
    p_fake = num_fake / len(y)
    p_real = num_real / len(y)
    
    # Initialize dictionaries to store the word counts for fake and real news articles
    fake_word_counts = Counter()
    real_word_counts = Counter()
    
    # Count the number of occurrences of each word in fake and real news articles
    for i in range(len(X)):
        if y[i] == 1:  # News article is fake
            fake_word_counts.update(X[i])
        else:  # News article is real
            real_word_counts.update(X[i])
    
    # Calculate the probability of each word appearing in a fake or real news article
    # Add 1 to the numerator to perform Laplace smoothing
    fake_word_probs = {word: (count + 1) / (num_fake + len(fake_word_counts)) for word, count in fake_word_counts.items()}
    real_word_probs = {word: (count + 1) / (num_real + len(real_word_counts)) for word, count in real_word_counts.items()}
    
    return p_fake, p_real, fake_word_probs, real_word_probs

def predict_fake_news(X, p_fake, p_real, fake_word_probs, real_word_probs):
    # Convert the test data to a sparse matrix
    X_sparse = csr_matrix(X)
    
    # Calculate the probability of each news article being fake or real
    p_fake_articles = p_fake * np.prod(np.power(fake_word_probs, X_sparse), axis=1)
    p_real_articles = p_real * np.prod(np.power(real_word_probs, X_sparse), axis=1)
    
    # Predict the class labels of the news articles
    y_pred = (p_fake_articles > p_real_articles).astype(int)
    
    return y_pred


# In[33]:


# Train the Naive Bayes classifier
p_fake, p_real, fake_word_probs, real_word_probs = train_naive_bayes(x_train_bow.toarray(), y_train)

# Predict the class labels of the test set
y_pred = predict_fake_news(x_test_bow.toarray(), p_fake, p_real, fake_word_probs, real_word_probs)


# In[ ]:


# Evaluate the performance of the classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:




