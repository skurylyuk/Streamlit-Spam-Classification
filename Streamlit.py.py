#!/usr/bin/env python
# coding: utf-8

# In[6]:


import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st


# In[7]:


st.write("# Spam Detection Engine")


# In[8]:


streamlit runstreamlit_app.py

