#!/usr/bin/env python
# coding: utf-8

# # Building a Simple Chatbot in Python using NLTK
# 
# The main technology which helps with the functioning of the chatbot is NLP(Natural Language Processing)

# # NLP
# 
# NLP is a way for computers to analyze, understand, and derive meaning from human language in a smart and useful way. By utilizing NLP, developers can organize and structure knowledge to perform tasks such as automatic summarization, translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, and topic segmentation.

# # Importing the Necessary Libraries

# In[1]:


import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


# # Downloading and Installing NLTK
# 
# NLTK(Natural Language Toolkit) is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries.

# In[2]:


pip install nltk


# In[3]:


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')


# # Corpus
# 
# I used Wikipedia pages for chatbots as my corpus. Copy the contents from the page and place it in a text file named ‘chatbot.txt’. However, you can use any corpus of your choice.

# In[4]:


f = open('chatbot.txt','r',errors = 'ignore')
raw = f.read()
raw = raw.lower()


# # Tokenisation

# In[5]:


sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)


# # Pre-Processing
# 
# We shall now define a function called LemTokens which will take as input the tokens and return normalized tokens.

# In[6]:


lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# # Keyword Matching
# 
# We shall define a function for a greeting by the bot i.e if a user’s input is a greeting, the bot shall return a response. ELIZA uses a simple keyword matching for greetings. We will utilize the same concept here.

# In[7]:


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","yo")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# # Generating a Response
# 
# We define a function response that searches the user’s input for one or more programmed keywords and returns one of several possible responses. If it doesn’t find the input matching any of the keywords, it returns a response: “I don’t understand you”

# In[8]:


def response(user_response):
    spar_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        spar_response=spar_response+"I don’t understand you"
        return spar_response
    else:
        spar_response = spar_response+sent_tokens[idx]
        return spar_response


# Now, we will command statements that we want the Bot to say while starting and ending a conversation upon the user’s input.

# In[9]:


flag=True
print("Spar: My name is Spar. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Spar: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("Spar: " + greeting(user_response))
            else:
                print("Spar: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False


# In[ ]:




