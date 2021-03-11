#!/usr/bin/env python
# coding: utf-8

# # Understanding the Problem Statement and Business Case
# 
# - We live in a world where we are constantly bombarded with social media feeds, tweets, and news articles. 
# - This huge data could be leveraged to predict people sentiment towards a particular company or stock.
# - Natural language processing (NLP) works by converting words (text) into numbers. These number are then used to train an AI/ML model to make predictions.
# - AI/ML based sentiment analysis models, can be used to understand the sentiment from public tweets, which could be used as a factor while making a buy/sell decision of securities.
# 
# # Import Libraries/Datasets and Performed Exploratory Data Analysis

# In[1]:


# import key libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Gensim is an open-source library for unsupervised topic modeling and natural language processing
# Gensim is implemented in Python and Cython.
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import plotly.express as px

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from transformers import pipeline


# In[2]:


# Loaded the stock news data
stock_df = pd.read_csv("D:\Python and Machine Learning for Financial Analysis\stock_sentiment.csv")


# In[3]:


# Let's view the dataset 
stock_df


# In[4]:


# dataframe information
stock_df.info()


# In[5]:


# checked for null values
stock_df.isnull().sum()


# In[6]:


sns.countplot(stock_df['Sentiment'])


# In[7]:


# Found the number of unique values in a particular column
stock_df['Sentiment'].nunique()


# # Performed Data Cleaning (Removed Punctuations from Text)

# In[8]:


string.punctuation


# In[9]:


Test = '$I love AI & Machine learning!!'
Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# In[10]:


Test = 'Good morning beautiful people :)... #I am having fun learning Finance with Python!!'


# In[11]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[12]:


# Joined the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# In[13]:


# Let's define a function to remove punctuations
def remove_punc(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)

    return Test_punc_removed_join


# In[14]:


# Let's remove punctuations from our dataset 
stock_df['Text Without Punctuation'] = stock_df['Text'].apply(remove_punc)


# In[15]:


stock_df


# In[16]:


stock_df['Text'][2]


# In[17]:


stock_df['Text Without Punctuation'][2]


# # Removed Punctuations using a Different Method

# In[18]:


Test_punc_removed = []
for char in Test: 
    if char not in string.punctuation:
        Test_punc_removed.append(char)

# Joined the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# # Performed Data Cleaning (Removed Stopwords)

# In[19]:


# downloaded stopwords
nltk.download("stopwords")
stopwords.words('english')


# In[20]:


# Obtained additional stopwords from nltk
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year'])
# stop_words.extend(['from', 'subject', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year', 'https'])


# In[21]:


# Removed stopwords and remove short words (less than 2 characters)
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) >= 3 and token not in stop_words:
            result.append(token)
            
    return result


# In[22]:


# Applied pre-processing to the text column
stock_df['Text Without Punc & Stopwords'] = stock_df['Text Without Punctuation'].apply(preprocess)


# In[23]:


stock_df['Text'][0]


# In[24]:


stock_df['Text Without Punc & Stopwords'][0]


# In[25]:


# Joined the words into a string
#stock_df['Processed Text 2'] = stock_df['Processed Text 2'].apply(lambda x: " ".join(x))


# In[26]:


stock_df


# In[27]:


# Obtained additional stopwords from nltk
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year', 'https'])


# In[28]:


# Removed stopwords and remove words with 2 or less characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >= 2 and token not in stop_words:
            result.append(token)
            
    return result


# In[29]:


# Applied pre-processing to the text column
stock_df['Text Without Punc & Stopwords'] = stock_df['Text Without Punctuation'].apply(preprocess)


# In[30]:


stock_df['Text'][0]


# In[31]:


stock_df['Text Without Punc & Stopwords'][0]


# In[32]:


stock_df


# # Plotted Wordcloud

# In[33]:


# Joined the words into a string
stock_df['Text Without Punc & Stopwords Joined'] = stock_df['Text Without Punc & Stopwords'].apply(lambda x: " ".join(x))


# In[34]:


# Plotted the word cloud for text with positive sentiment
plt.figure(figsize = (20, 20)) 
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800).generate(" ".join(stock_df[stock_df['Sentiment'] == 1]['Text Without Punc & Stopwords Joined']))
plt.imshow(wc, interpolation = 'bilinear');


# # Visualized the Wordcloud for Tweets that have Negative Sentiment

# In[35]:


# Plotted the word cloud for text that is negative
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 1000, width = 1600, height = 800 ).generate(" ".join(stock_df[stock_df['Sentiment'] == 0]['Text Without Punc & Stopwords Joined']))
plt.imshow(wc, interpolation = 'bilinear');


# # Visualized Cleaned Datasets

# In[36]:


stock_df


# In[37]:


nltk.download('punkt')


# In[38]:


# word_tokenize is used to break up a string into words
print(stock_df['Text Without Punc & Stopwords Joined'][0])
print(nltk.word_tokenize(stock_df['Text Without Punc & Stopwords Joined'][0]))


# In[39]:


# Obtained the maximum length of data in the document
# This will be later used when word embeddings are generated
maxlen = -1
for doc in stock_df['Text Without Punc & Stopwords Joined']:
    tokens = nltk.word_tokenize(doc)
    if(maxlen < len(tokens)):
        maxlen = len(tokens)
print("The maximum number of words in any document is:", maxlen)


# In[40]:


tweets_length = [ len(nltk.word_tokenize(x)) for x in stock_df['Text Without Punc & Stopwords Joined'] ]
tweets_length


# In[41]:


# Plotted the distribution for the number of words in a text
fig = px.histogram(x = tweets_length, nbins = 50)
fig.show()


# In[42]:


# Plotted the word count
sns.countplot(stock_df['Sentiment'])


# # Prepared the Data by Tokenizing and Padding
# 
# ## Tokenizer
# 
# - Tokenizer allows us to vectorize text corpus.
# - Tokenization works by turning each text into a sequence of integers.

# In[43]:


stock_df


# In[44]:


# Obtained the total words present in the dataset
list_of_words = []
for i in stock_df['Text Without Punc & Stopwords']:
    for j in i:
        list_of_words.append(j)


# In[45]:


list_of_words


# In[46]:


# Obtained the total number of unique words
total_words = len(list(set(list_of_words)))
total_words


# In[47]:


# Splitted the data into test and train 
X = stock_df['Text Without Punc & Stopwords']
y = stock_df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# In[48]:


X_train.shape


# In[49]:


X_test.shape


# In[50]:


X_train


# In[51]:


# Created a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(X_train)

# Training data
train_sequences = tokenizer.texts_to_sequences(X_train)

# Testing data
test_sequences = tokenizer.texts_to_sequences(X_test)


# In[52]:


train_sequences


# In[53]:


test_sequences


# In[54]:


print("The encoding for document\n", X_train[1:2],"\n is: ", train_sequences[1])


# In[55]:


# Added padding to training and testing
padded_train = pad_sequences(train_sequences, maxlen = 29, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences, maxlen = 29, truncating = 'post')


# In[56]:


for i, doc in enumerate(padded_train[:3]):
     print("The padded encoding for document:", i+1," is:", doc)


# In[57]:


# Converted the data to categorical 2D representation
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)


# In[58]:


y_train_cat.shape


# In[59]:


y_test_cat.shape


# In[60]:


y_train_cat


# In[61]:


# Added padding to training and testing
padded_train = pad_sequences(train_sequences, maxlen = 15, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences, maxlen = 15, truncating = 'post')


# In[62]:


for i, doc in enumerate(padded_train[:3]):
     print("The padded encoding for document:", i+1," is:", doc)


# In[63]:


# Converted the data to categorical 2D representation
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)


# In[64]:


y_train_cat.shape


# In[65]:


y_test_cat.shape


# In[66]:


y_train_cat


# # Understanding the Theory and Intuition behind Recurrent Neural Networks and Long Short Term Memory Networks (LSTM)
# 
# ## Introduction to Recurrent Neural Networks (RNN)
# 
# - Feedforward Neural Networks (vanilla networks) map a fixed size input (such as image) to a fixed size output (classes or probabilities).
# - A drawback in Feedforward networks is that they do not have any time dependency or memory effect.
# - A RNN is a type of ANN that is designed to take temporal dimension into consideration by having a memory (internal state) (feedback loop).
# - A RNN contains a temporal loop in which the hidden layer not only gives an output but it feeds itself as well.
# - An extra dimension is added which is time!
# - RNN can recall what happened in the previous time stamp so it works great with sequence of text.
# 
# ## Long Short Term Memory Networks
# 
# - LSTM contains gates that can allow or block information from passing by.
# - Gates consist of a sigmoid neural net layer along with a pointwise multiplication operation.
# + Sigmoid output ranges from 0 to 1:
#     - 0 = Don't allow any data to flow
#     - 1 = Allow everything to flow!
#     
# # Build a Custom-Based Deep Neural Network to Perform Sentiment Analysis
# 
# ## Embedding Layer
# 
# - Embedding layers learn low-dimensional continuous representation of discrete input variables.
# - For example, let say we have 100000 unique values in our data and want to train the model with these data. Though we can use these as such, it would require more resources to train. With embedding layer, you can specify the number of low-dimensional feature that you would need to represent the input data, in this case let's take the value to be 200.
# - Now, what happens is embedding layer learns the way to represent 100000 variables with 200 variables, similar to Principal Component Analysis (PCA) or Autoencoder.
# - This in-turn helps the subsequent layers to learn more effectively.

# In[67]:


# Sequential Model
model = Sequential()

# embedding layer
model.add(Embedding(total_words, output_dim = 512))

# Bi-Directional RNN and LSTM
model.add(LSTM(256))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()


# In[68]:


# train the model
model.fit(padded_train, y_train_cat, batch_size = 32, validation_split = 0.2, epochs = 2)


# # Trained the Model using Different Embedding Output Dimension

# In[69]:


#model = Sequential()

# embedding layer
#model.add(Embedding(total_words, output_dim = 256))

# Bi-Directional RNN and LSTM
#model.add(Bidirectional(LSTM(128)))

# Dense layers
#model.add(Dense(128, activation = 'relu'))
#model.add(Dense(1,activation = 'sigmoid'))
#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
#model.summary()


# # Assessed Trained Model Performance

# In[70]:


# Made prediction
pred = model.predict(padded_test)


# In[71]:


# Made prediction
prediction = []
for i in pred:
  prediction.append(np.argmax(i))


# In[72]:


# list containing original values
original = []
for i in y_test_cat:
  original.append(np.argmax(i))


# In[73]:


# Accuracy score on text data
accuracy = accuracy_score(original, prediction)
accuracy


# In[74]:


# Plotted the confusion matrix
cm = confusion_matrix(original, prediction)
sns.heatmap(cm, annot = True)


# In[75]:


# Used pipeline from transformer to perform specific task. 
# Mentioned sentiment analysis as task and passed in the string to it, to get the results
# We can specify tasks like topic modeling, Q and A, text summarization here.

#nlp = pipeline('sentiment-analysis')

# Made prediction on the test data
#pred = nlp(list(X_test))

# Since predicted value is a dictionary, get the label from the dict
#prediction = []
#for i in pred:
#  prediction.append(i['label'])

# print the final results
#for i in range(len(prediction[:3])):
#  print("\n\nNews :\n\n", df[df.combined == X_test.values[i]].Text.item(), "\n\nOriginal value :\n\n",
#      category[df[df.combined == X_test.values[i]].Sentiment.item()], "\n\nPredicted value :\n\n", prediction[i], "\n\n\n")

