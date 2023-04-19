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


# In[ ]:





# In[31]:


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


# In[32]:


x = final_data['text']
y = final_data['label']


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[34]:


# Transform the training data into bag of words features using the CountVectorizer
count_vectorizer = CountVectorizer()
x_train_bow = count_vectorizer.fit_transform(x_train)


# In[35]:


# Transform the test data into bag of words features using the CountVectorizer
x_test_bow = count_vectorizer.transform(x_test)


# In[36]:


# Get the feature names of 'count_vectorizer'

print(count_vectorizer.get_feature_names()[:10])


# In[37]:


nbc_pipeline = Pipeline([
        ('NBCV',count_vectorizer),
        ('nb_clf',MultinomialNB())])
nbc_pipeline.fit(x_train,y_train)


# In[38]:


predicted_nbc = nbc_pipeline.predict(x_test)
score = metrics.accuracy_score(y_test, predicted_nbc)
print(f'Accuracy: {round(score*100,2)}%')


# In[39]:


cm1 = metrics.confusion_matrix(y_test, predicted_nbc, labels=['Fake', 'True'])
plot_confusion_matrix(cm1, classes=['Fake', 'True'])


# In[40]:


print(cm1)


# In[41]:


print(metrics.classification_report(y_test, predicted_nbc))


# In[42]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
xv_train = tfidf_vectorizer.fit_transform(x_train)
xv_test = tfidf_vectorizer.transform(x_test)


# In[43]:


# get feature names

# Get the feature names of 'tfidf_vectorizer'

print(tfidf_vectorizer.get_feature_names()[-10:])


# In[44]:


# building classifier using naive bayes 
# Naive Bayes classifier for Multinomial model

nb_pipeline = Pipeline([
        ('NBTV',tfidf_vectorizer),
        ('nb_clf',MultinomialNB())])


# In[45]:


# Fit Naive Bayes classifier according to X, y

nb_pipeline.fit(x_train,y_train)


# In[46]:


# Perform classification on an array of test vectors X

predicted_nbt = nb_pipeline.predict(x_test)


# In[47]:


score = metrics.accuracy_score(y_test, predicted_nbt)
print(f'Accuracy: {round(score*100,2)}%')


# In[48]:


cm = metrics.confusion_matrix(y_test, predicted_nbt, labels=['Fake', 'True'])
plot_confusion_matrix(cm, classes=['Fake', 'True'])


# In[49]:


print(cm)


# In[50]:


print(metrics.classification_report(y_test, predicted_nbt))


# In[53]:


# saving best model to the disk
import pickle
model_file = 'final_model.pkl'
pickle.dump(nbc_pipeline,open(model_file,'wb'))


# In[54]:


model_vector = 'final_vector.pkl'
pickle.dump(count_vectorizer,open(model_vector,'wb'))


# In[ ]:


var = input("Please enter the news text you want to verify: ")

# function to run for prediction
def detecting_fake_news(var):  
    
    #retrieving the best model for prediction call
    load_model = pickle.load(open('final_model.sav', 'rb'))
    prediction = load_model.predict([var])

    return (print("The given statement is ",prediction[0]))

if __name__ == '__main__':
    detecting_fake_news(var)


# In[ ]:




