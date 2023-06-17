#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv('Documents/Email_spam_classifier/mail_data.csv')


# In[3]:


print(df)


# In[4]:


data=df.where((pd.notnull(df)),'')


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1


# In[9]:


X=data['Message']
Y=data['Category']


# In[10]:


print(X)


# In[11]:


print(Y)


# In[12]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)


# In[13]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[14]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[15]:


feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# In[16]:


print(X_train)


# In[17]:


print(X_train_features)


# In[18]:


model=LogisticRegression()


# In[19]:


model.fit(X_train_features,Y_train)


# In[20]:


prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)


# In[21]:


print('Accuracy on training data: ',accuracy_on_training_data)


# In[23]:


prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)


# In[24]:


print('Accuracy on test data: ',accuracy_on_test_data)


# In[30]:


input_your_mail=["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
input_data_features=feature_extraction.transform(input_your_mail)
prediction=model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
    print("Hae Mail")
else:
    print('Spam Mail')


# In[ ]:





# In[ ]:




