#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


csv = pd.read_csv('Iris.csv')


# In[3]:


csv


# In[4]:


csv.info()


# In[5]:


csv.isnull().sum()


# In[7]:


csv.describe()


# In[10]:


sns.pairplot(csv, hue='Species')


# In[12]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
csv['species_get'] = label_encoder.fit_transform(csv['Species'])


# In[14]:


sns.heatmap(csv.drop(columns=['Id','Species']).corr(), annot=True, cmap='coolwarm')
plt.show()


# In[15]:


iris = load_iris()
X = iris.data
y = iris.target


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[17]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[18]:


knn.fit(X_train, y_train)


# In[19]:


y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[20]:


sns.pairplot(data=sns.load_dataset("iris"), hue="species")
plt.title("Iris Dataset")
plt.show()


# In[ ]:





# In[ ]:




