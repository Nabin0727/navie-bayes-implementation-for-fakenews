#!/usr/bin/env python
# coding: utf-8

# In[45]:


from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string


# In[46]:


app = Flask(__name__)


# In[47]:


def predict_fake_news(X, p_fake, p_real, fake_word_probs, real_word_probs, alpha=1):
    y_pred = []
    num_fake = 0
    num_real = 0
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
    
    return np.array(y_pred)


# In[48]:


# Load model
with open('naive_bayes_model.pkl', 'rb') as file:
    model_train = pickle.load(file)


# In[ ]:





# In[49]:


# Load the vecrotizer 
with open('tfidf_vectorizer.pkl', 'rb') as file:
    model_vectorizer = pickle.load(file)


# In[50]:


translator = str.maketrans('', '', string.punctuation)


# In[61]:


def fake_news_det(news):
    predictions = []
    for item in news:
        input_data = item.lower()
        input_data = input_data.translate(translator)
        stop_words = stopwords.words('english')
        
        input_data = ' '.join([word for word in input_data.split() if word not in (stop_words)])
        
        vectorized_input_data = model_vectorizer.transform([input_data])
        
        # Predict the class label of the new data
        prediction = predict_fake_news(vectorized_input_data, model_train["p_fake"], model_train["p_real"], 
                            model_train["fake_word_probs"], model_train["real_word_probs"])
        #prediction = model.predict(vectorized_input_data)
        
        predictions.append(prediction)
    
    return predictions


# In[63]:


#Flask routing 
@app.route("/")

def home():
    return render_template("Home.html")


# In[68]:


@app.route('/predict', methods = ['POST','GET'])
def predict_fake():
    if request.method == 'POST':
        news = request.form['message']
        pred = fake_news_det(news)[0]
        print(pred)
        print(str(pred))
        return render_template('Home.html', prediction = pred, news=news)
    else:
        return render_template('Home.html')


# In[69]:


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port='8098')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




