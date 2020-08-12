#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


heart=pd.read_csv('iris.csv')


# In[3]:


heart.head()


# In[7]:


y=heart[['species']]


# In[8]:


x=heart[['sepal_length']]


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[14]:


heart.shape


# In[12]:


y_train.shape


# In[18]:


from sklearn.tree import DecisionTreeClassifier


# In[19]:


dt=DecisionTreeClassifier()


# In[20]:


dt.fit(x_train,y_train)


# In[21]:


y_pred=dt.predict(x_test)


# In[22]:


y_test.head(),y_pred[:5]


# In[23]:


from sklearn.metrics import confusion_matrix


# In[24]:


confusion_matrix(y_test,y_pred)


# In[25]:


#1st row for setosa,2nd for versicolor and 3rd for verginica 
#13, 10 and 6 are correctly classified


# In[26]:


#Accuracy
(13+10+9)/(13+10+9+1+2+6+4)


# In[27]:


import matplotlib.pyplot as plt


# In[29]:


#plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.show()


# In[ ]:


#decision tree regressor


# In[31]:


heart.head()


# In[32]:


y=heart[['sepal_length']]


# In[33]:


x=heart[['petal_length']]


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[36]:


from sklearn.tree import DecisionTreeRegressor


# In[38]:


dt2=DecisionTreeRegressor()


# In[39]:


dt2.fit(x_train,y_train)


# In[41]:


y_pred=dt2.predict(x_test)
y_test.head(),y_pred[:5]


# In[43]:


from sklearn.metrics import mean_squared_error


# In[44]:


mean_squared_error(y_test,y_pred)


# In[45]:


import matplotlib.pyplot as plt


# In[47]:


plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,color='red')
plt.show()

