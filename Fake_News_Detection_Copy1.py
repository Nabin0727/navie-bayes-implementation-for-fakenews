#!/usr/bin/env python
# coding: utf-8

# In[74]:


# Including Libraries
# necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
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


# In[75]:


#We will import the file through the pandas 
TrueNews = pd.read_csv("True.csv")
FakeNews = pd.read_csv("Fake.csv")
#Let's check our data
print(TrueNews.head(5))
#print(FakeNews.head(5))


# In[76]:


#Checking shape for both files
print(TrueNews.shape)
print(FakeNews.shape)


# In[77]:


print('FAKE',FakeNews.isnull().sum())
print('TRUE',TrueNews.isnull().sum())


# In[78]:


#Columns Print
print(list(TrueNews.columns))
print(list(FakeNews.columns))


# In[79]:


#We are adding label fake and true
TrueNews['label'] = 'True'
TrueNews.head(5)


# In[80]:


FakeNews['label'] = 'Fake'
FakeNews.head(5)


# In[81]:


TrueNews.head()


# In[82]:


#Let's concatenate the dataframes
frames = [TrueNews, FakeNews]
news_dataset = pd.concat(frames)
news_dataset


# In[ ]:





# In[83]:


#New combined dataset 
news_dataset.describe()


# In[84]:


news_dataset.info()


# In[85]:


final_data = news_dataset.dropna()


# In[86]:


final_data.isnull().sum()


# In[87]:


# Removing the date 
final_data.drop(["date"],axis=1,inplace=True)
final_data.head()


# In[88]:


# Removing the title
final_data.drop(["title"],axis=1,inplace=True)
final_data.head()


# In[89]:


#First lets convert our data into lower case 
final_data['text'] = final_data['text'].apply(lambda x: x.lower())
#final_data['title'] = final_data['title'].apply(lambda x: x.lower())
final_data.head()


# In[90]:


# Drop odd rows
#final_data = final_data.iloc[::2]


# In[91]:


#New combined dataset 
final_data.describe()


# In[92]:


# Drop odd rows
#final_data = final_data.iloc[::2]


# In[93]:


#New combined dataset 
final_data.describe()


# In[94]:


#Removing punctuation
import string

def remove_punctuation(text):
    #all_list = [char for char in text if char not in string.punctuation]
    #no_punct = ''.join(all_list)
    translator = str.maketrans('', '',string.punctuation)
    no_punct = text.translate(translator)
    return no_punct

final_data['text'] = final_data['text'].apply(remove_punctuation)


# In[95]:


# Verifying
final_data.head()


# In[96]:


# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

final_data['text'] = final_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[97]:


final_data.head()


# In[98]:


final_data = final_data.sample(frac = 1)


# In[99]:


final_data.head()


# In[100]:


final_data.describe()


# In[101]:


final_data.to_csv('final_data.csv')


# In[102]:


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


# In[103]:


x = final_data['text']
y = final_data['label']


# In[104]:


# Convert the class labels from strings to integers
y = np.array([1 if label == "True" else 0 for label in y])


# In[105]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[106]:


# Transform the training data into bag of words features using the CountVectorizer
count_vectorizer = CountVectorizer()

x_train_bow = count_vectorizer.fit_transform(x_train)


# In[107]:


# Transform the test data into bag of words features using the CountVectorizer

x_test_bow = count_vectorizer.transform(x_test)


# In[108]:


# Get the feature names of 'count_vectorizer'

print(count_vectorizer.get_feature_names()[:10])


# In[137]:


from collections import defaultdict
from scipy.sparse import csr_matrix

def train_naive_bayes(X, y, alpha=1.0):
    # Convert X to a sparse matrix
    X_sparse = csr_matrix(X)
    
    # Calculate the number of fake and real news articles in the training set
    num_fake = y.sum()
    num_real = len(y) - num_fake
    
    # Calculate the probability of a news article being fake or real
    p_fake = (num_fake + alpha) / (len(y) + 2 * alpha)
    p_real = (num_real + alpha) / (len(y) + 2 * alpha)
    
    # Calculate the probability of each word appearing in a fake or real news article
    fake_word_probs = defaultdict(int)
    real_word_probs = defaultdict(int)
    for i in range(X_sparse.shape[0]):
        article = X_sparse[i, :]
        words = article.indices
        for index in words:
            if y[i] == 1:  # News article is fake
                fake_word_probs[index] += 1
            else:  # News article is real
                real_word_probs[index] += 1
    
    # Normalize the word counts to obtain probabilities
    for word in fake_word_probs:
        fake_word_probs[word] = (fake_word_probs[word] + alpha) / (num_fake + 2 * alpha)
    for word in real_word_probs:
        real_word_probs[word] = (real_word_probs[word] + alpha) / (num_real + 2 * alpha)
    
    # Save the trained model using pickle
    model = {"p_fake": p_fake, "p_real": p_real, "fake_word_probs": fake_word_probs, "real_word_probs": real_word_probs}
    with open("naive_bayes_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    return p_fake, p_real, fake_word_probs, real_word_probs


def predict_fake_news(X, p_fake, p_real, fake_word_probs, real_word_probs, alpha=1):
    y_pred = []
    num_fake = 0
    num_real = 0
    accuracy = []
    for i in range(X.shape[0]):
        article = X[i, :]
        p_real_article = 1.0
        p_fake_article = 1.0
        words = article.nonzero()[1]
        for index in words:
            if index in fake_word_probs:
                p_fake_article *= fake_word_probs[index]
            else:
                p_fake_article *= alpha / (num_fake + 2 * alpha)
            if index in real_word_probs:
                p_real_article *= real_word_probs[index]
            else:
                p_real_article *= alpha / (num_real + 2 * alpha)
        
        # Predict the class label of the news article
        if p_fake_article * p_fake > p_real_article * p_real:
            y_pred.append(1)
        else:
            y_pred.append(0)
        
        if y_pred[-1] == 1:
            num_fake += 1
        else:
            num_real += 1
            
    #accuracy = num_correct / X.shape[0]
    return np.array(y_pred)

# Saving the predict fake news
with open('predict_fake_news.pkl', 'wb') as f:
    pickle.dump(predict_fake_news, f)


# In[138]:


# Train the Naive Bayes classifier
p_fake, p_real, fake_word_probs, real_word_probs = train_naive_bayes(x_train_bow, y_train)


# In[ ]:





# In[139]:


# Predict the class labels of the test set
y_pred = predict_fake_news(x_test_bow, p_fake, p_real, fake_word_probs, real_word_probs)


# In[140]:


print(y_pred)


# In[141]:


# Evaluate the performance of the classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[142]:


# Accuracy count
accuracy_count = accuracy_score(y_test, y_pred)


# In[143]:


np.unique(y_test)


# In[144]:


cm1 = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
plot_confusion_matrix(cm1, classes=['Fake', 'True'])


# In[145]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# calculate confusion matrix
cm = metrics.confusion_matrix(y_test, tfidf_pred, labels=[0, 1], normalize='true')

# plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(['0', '1']))
plt.xticks(tick_marks, ['Fake', 'True'], rotation=45)
plt.yticks(tick_marks, ['Fake', 'True'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# add labels to cells
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, '{:.2f}'.format(cm[i, j]),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.show()


# In[146]:


print(cm1)


# In[147]:


print(metrics.classification_report(y_test, y_pred))


# In[148]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


# In[149]:


# get feature names

# Get the feature names of 'tfidf_vectorizer'

print(tfidf_vectorizer.get_feature_names()[-10:])


# In[150]:


# Train the Naive Bayes classifier
p_fake, p_real, fake_word_probs, real_word_probs = train_naive_bayes(tfidf_train, y_train)


# In[151]:


# Predict the class labels of the test set
tfidf_pred = predict_fake_news(tfidf_test, p_fake, p_real, fake_word_probs, real_word_probs)


# In[152]:


# Evaluate the performance of the classifier
accuracy = np.mean(tfidf_pred == y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[153]:


# Accuracy count
accuracy_tfidf = accuracy_score(y_test, tfidf_pred)


# In[163]:


print(accuracy_tfidf)


# In[154]:


cm1 = metrics.confusion_matrix(y_test, tfidf_pred, labels=[0, 1])
plot_confusion_matrix(cm1, classes=['Fake', 'True'])


# In[ ]:





# In[155]:


print(cm1)


# In[156]:


print(metrics.classification_report(y_test, tfidf_pred))


# In[162]:


# Plot bar chart
accuracies = [accuracy_count, accuracy_tfidf]
labels = ['CountVectorizer', 'TfidfVectorizer']

# Plotting the accuracy values for count vectorizer
plt.plot(accuracy_count, color='blue', label='Count Vectorizer')

# Plotting the accuracy values for tfidf vectorizer
plt.plot(accuracy_tfidf, color='green', label='TFIDF Vectorizer')

# Adding plot labels and legend
plt.title('Accuracy Comparison')
plt.xlabel('Test Set')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[52]:


# Save the tfidf_vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)


# In[53]:


var = input("Please enter the news text you want to verify: ")
# function to run for prediction
def detecting_fake_news(var):  
    #retrieving the best model for prediction call
    # Load the saved model
    with open('naive_bayes_model.pkl', 'rb') as file:
        model_train = pickle.load(file)
        
    # Load the vecrotizer 
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        model_vectorizer = pickle.load(file)
  
    # Convert the input to a 2-dimensional array with shape (1, num_features)
    X = model_vectorizer.transform([var])
    
    # Predict the class label of the new data
    prediction = predict_fake_news(X, model_train["p_fake"], model_train["p_real"], 
                            model_train["fake_word_probs"], model_train["real_word_probs"])

    return (print("The given statement is ",prediction[0]))

if __name__ == '__main__':
    detecting_fake_news(var)

