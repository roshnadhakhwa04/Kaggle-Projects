#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd


# In[57]:


heart=pd.read_csv('heart.csv')


# In[58]:


heart.head()


# In[59]:


y=heart[['target']]


# In[60]:


x=heart[['age','sex','cp']]


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[63]:


from sklearn.linear_model import LinearRegression


# In[64]:


lr=LinearRegression()


# In[66]:


lr.fit(x_train,y_train)


# In[67]:


y_pred=lr.predict(x_test)


# In[68]:


y_test.head(),y_pred[:5]


# In[74]:


from sklearn.metrics import mean_squared_error


# In[75]:


mean_squared_error(y_test,y_pred)


# In[80]:


import matplotlib.pyplot as plt


# In[81]:


plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.show()

