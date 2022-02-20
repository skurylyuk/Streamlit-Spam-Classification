#!/usr/bin/env python
# coding: utf-8

# In[20]:


# import important modules
import numpy as np
import pandas as pd

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB # classifier 

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    plot_confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# text preprocessing modules
from string import punctuation 

# text preprocessing modules
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re #regular expression

# Download dependency
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
):
    nltk.download(dependency)
    
import warnings
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)
from sklearn.decomposition import NMF


# In[21]:


df = pd.read_csv('SPAM text message 20170820 - Data.csv', encoding="latin-1")


# In[22]:


df = df.drop_duplicates()
df = df.reset_index(inplace = False)[['Category','Message']]
print(df.shape)


# In[23]:


df['labels'] = df['Category'].map({'ham':0, 'spam':1})
df


# In[24]:


spam_messages= df.loc[df.Category=="spam"]["Message"]
not_spam_messages= df.loc[df.Category=="ham"]["Message"]

print("spam count: " +str(len(df.loc[df.Category=="spam"])))
print("not spam count: " +str(len(df.loc[df.Category=="ham"])))

not_spam_messages


# # Text Preprocessing 

# In[25]:


def preprocessing_text(texts):
    df["Clean_Message"] = df["Message"].str.lower() #puts everything in lowercase
    df["Clean_Message"] = df["Message"].replace(r'http\S+', '', regex=True) # removing any links 
    df["Clean_Message"] = df["Message"].replace(r'www.[^ ]+', '', regex=True)
    df["Clean_Message"] = df["Message"].replace(r'[0-9]+', " ", regex = True) #removing numbers
    df["Clean_Message"] = df["Message"].replace (r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', '', regex=True) #remove special characters and puntuation marks 
    df["Clean_Message"] = df["Message"].replace(r"[^A-Za-z]", " ", regex = True) #replace any item that is not a letter
    

    return texts


# In[26]:


texts = preprocessing_text(df)
texts


# In[27]:


stop_words =  stopwords.words('english')

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
    text = text.lower()
    
        
    #Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)


# In[28]:


#clean the review
df["Clean_Message_2"]= df["Message"].apply(text_cleaning)


# In[29]:


df


# # Modeling - TFIDF Vectorization
# 

# In[30]:


#TFIDF 

docs = df.Clean_Message_2
tfidf= TfidfVectorizer(stop_words= "english",
                       max_df = .4, 
                       min_df = 5, #maybe 5 or 6
                       max_features = 20000,
                       lowercase=True, 
                       analyzer='word',
                       ngram_range=(1,3),
                       dtype=np.float32)
doc_term_matrix = tfidf.fit_transform(docs) #should this be values?

#


# In[31]:


X= df['Clean_Message_2']
y= df['labels']


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42, shuffle = True, stratify = y)


# In[33]:


from sklearn.neural_network import MLPClassifier
neural_net_pipeline = Pipeline([('vectorizer', tfidf), 
                                ('nn', MLPClassifier(hidden_layer_sizes=(700, 700)))])


# In[36]:


neural_net_pipeline.fit(X_train, y_train)


# In[37]:


# Testing the Pipeline

y_pred = neural_net_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print('Accuracy: {} %'.format(100 * accuracy_score(y_test, y_pred)))


# In[38]:


from joblib import dump
dump(neural_net_pipeline, 'spam_classifier.joblib')

