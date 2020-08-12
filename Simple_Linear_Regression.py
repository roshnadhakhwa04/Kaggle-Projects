#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


heart=pd.read_csv('iris.csv')


# In[3]:


heart.head()


# In[4]:


y=heart[['sepal_length']]


# In[5]:


x=heart[['petal_length']]


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


lr=LinearRegression()


# In[10]:


lr.fit(x_train,y_train)


# In[11]:


y_pred=lr.predict(x_test)


# In[12]:


y_test.head(),y_pred[:4]


# In[13]:


from sklearn.metrics import mean_squared_error


# In[14]:


mean_squared_error(y_test,y_pred)


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.show()


# In[18]:


from sklearn.metrics import mean_absolute_error


# In[20]:


mean_absolute_error(y_test,y_pred)

