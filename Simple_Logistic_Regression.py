#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


heart=pd.read_csv('iris.csv')


# In[3]:


heart.head()


# In[21]:


y=heart[['species']]


# In[22]:


x=heart[['petal_length']]


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


lr=LogisticRegression()


# In[27]:


lr.fit(x_train,y_train)


# In[28]:


y_pred=lr.predict(x_test)


# In[30]:


y_test.head(),y_pred[:5]


# In[32]:


from sklearn.metrics import confusion_matrix


# In[33]:


confusion_matrix(y_test,y_pred)


# In[34]:


#Accuracy
(16+14+13)/(16+14+13+2)


# In[35]:


import matplotlib.pyplot as plt


# In[44]:



plt.plot(x_test,y_pred,color='red')
plt.show()

