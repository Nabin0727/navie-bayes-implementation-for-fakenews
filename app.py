#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string


# In[5]:


app = Flask(__name__)


# In[6]:


# Load model
with open('naive_bayes_model.pkl', 'rb') as file:
    model_train = pickle.load(file)


# In[7]:


# Load the vecrotizer 
with open('tfidf_vectorizer.pkl', 'rb') as file:
    model_vectorizer = pickle.load(file)


# In[8]:


translator = str.maketrans('', '', string.punctuation)


# In[9]:


def preprocess(article):
    article = article.lower()
    article = article.translate(translator)
    stop_words = stopwords.words('english')
    article = ' '.join([word for word in article.split() if word not in (stop_words)])
    return article


# In[10]:


def predict_fake_news(X, p_fake, p_real, fake_word_probs, real_word_probs, alpha=1):
    num_fake = 0
    num_real = 0
    y_pred = []
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
            print(p_fake_article * p_fake)
            print((p_real_article * p_real))
            print(1)
        else:
            y_pred.append(0)
        
        if y_pred[-1] == 1:
            num_fake += 1
        else:
            num_real += 1
    
    return y_pred


# In[11]:


#Flask routing 
@app.route("/")

def home():
    return render_template("Home.html")


# In[12]:


@app.route('/predict', methods = ['POST','GET'])
def predict_news():
    pred = []
    if request.method == 'POST':
        news = request.form['message']
        news = preprocess(news)
        vectorized_article = model_vectorizer.transform([news])
        prediction = predict_fake_news(vectorized_article, model_train["p_fake"], model_train["p_real"], 
                                model_train["fake_word_probs"], model_train["real_word_probs"])[0]
        pred.append(prediction)
        print(pred)

        # Evaluate the performance of the classifier
        
        return render_template('Home.html', prediction=pred)
    else:
        return render_template('Home.html')


# In[13]:


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port='9202')


# In[ ]:




