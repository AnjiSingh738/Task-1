#!/usr/bin/env python
# coding: utf-8

# 
# # GRIP: The Sparks Foundation

# # Data Science and Business Analytics Intern

# # Author: Anjali Singh

# # Task 1: Prediction Using Supervised ML

# In this task we have to predict the percentage score of a student based on the number of hours studied. The task has twovariables where the feature is the no. of hours studied and the target value is the percentage score. This can be solved using simple Linear Regression.
# 

# In[ ]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading data from remote url

# In[3]:


url="http://bit.ly/w-data"
data = pd.read_csv(url)


# # Exploring Data

# In[5]:


print(data.shape)
data.head()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.plot(kind="scatter",x="Hours",y="Scores")
plt.show()


# In[12]:


data.corr(method='pearson')


# In[13]:


data.corr(method='spearman')


# In[14]:


hours=data['Hours']
scores=data['Scores']


# In[15]:


sns.distplot(hours)


# In[16]:


sns.distplot(scores)


# # Linear Regression

# In[25]:


x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[26]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 50)


# In[28]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)


# In[30]:


m = reg.coef_
c = reg.intercept_
line = m*x + c
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[31]:


y_pred=reg.predict(x_test)


# In[35]:


actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted


# In[37]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# In[38]:


h = 9.25
s = reg.predict([[h]])
print("If a student for {} hours per day he/she will score {} % in exam.".format(h,s))


# # Model Evaluation 

# In[40]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
print('R2 Score:',r2_score(y_test, y_pred))


# In[ ]:




