#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle


# In[2]:


df = pd.read_csv("diabetes.csv")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


print("Number of zero values in Glucose = ",df[df["Glucose"] == 0].shape[0])
print("Number of zero values in BloodPressure = ",df[df["BloodPressure"] == 0].shape[0])
print("Number of zero values in SkinThickness = ",df[df["SkinThickness"] == 0].shape[0])
print("Number of zero values in Insulin = ",df[df["Insulin"] == 0].shape[0])
print("Number of zero values in BMI = ",df[df["BMI"] == 0].shape[0])
df["Glucose"] = df["Glucose"].replace(0,df["Glucose"].mean())


# In[7]:


df["BloodPressure"] = df["BloodPressure"].replace(0,df["BloodPressure"].mean())
df["SkinThickness"] = df["SkinThickness"].replace(0,df["SkinThickness"].mean())
df["Insulin"] = df["Insulin"].replace(0,df["Insulin"].mean())
df["BMI"] = df["BMI"].replace(0,df["BMI"].mean())


# In[8]:


df["Outcome"].value_counts()


# In[9]:


N, P = df["Outcome"].value_counts()
print("Diabetese Negative (0): ",N)
print("Diabetese Positive (1): ",P)
f , ax = plt.subplots(1,2,figsize = (10,5))
df["Outcome"].value_counts().plot.pie(autopct = "%1.1f%%", ax = ax[0], shadow = True)
ax[0].set_title("Outcome")
ax[0].set_ylabel(" ")
sns.countplot("Outcome", data = df)
ax[1].set_title("Outcome")
plt.grid()
plt.show()


# In[10]:


df.hist(bins = 10, figsize = (10,10))
plt.show()


# In[11]:


sns.pairplot(df, hue = "Outcome")
plt.show()


# In[12]:


plt.figure(figsize = (12,6))
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[13]:


fig, ax = plt.subplots(4,2, figsize=(20,25))
sns.distplot(df.Age, bins = 10, color = 'red', ax=ax[0,0]) 
sns.distplot(df.Pregnancies, bins = 10, color = 'orange', ax=ax[0,1]) 
sns.distplot(df.Glucose, bins = 10, color = 'blue', ax=ax[1,0]) 
sns.distplot(df.BloodPressure, bins = 10, color = 'black', ax=ax[1,1]) 
sns.distplot(df.SkinThickness, bins = 10, color = 'green', ax=ax[2,0])
sns.distplot(df.Insulin, bins = 10, color = 'purple', ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 10, color = 'pink', ax=ax[3,0]) 
sns.distplot(df.BMI, bins = 10, ax=ax[3,1])


# In[14]:


df.groupby("Outcome").mean()


# In[15]:


X = df.drop(columns = "Outcome", axis=1)
Y = df["Outcome"]


# In[16]:


X


# In[17]:


Y


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


Y_train.shape


# In[22]:


Y_test.shape


# In[23]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)


# In[25]:


from sklearn.tree import DecisionTreeClassifier
tr =  DecisionTreeClassifier()
tr.fit(X_train, Y_train)


# In[26]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)


# In[27]:


from sklearn.svm import SVC
sv = SVC()
sv.fit(X_train, Y_train)


# In[28]:


Y_pred1 = lr.predict(X_test)
Y_pred2 = knn.predict(X_test)
Y_pred3 = tr.predict(X_test)
Y_pred4 = rf.predict(X_test)
Y_pred5 = sv.predict(X_test)


# In[29]:


accuracy_score(Y_test, Y_pred1)


# In[30]:


accuracy_score(Y_test, Y_pred2)


# In[31]:


accuracy_score(Y_test, Y_pred4)


# In[32]:


accuracy_score(Y_test, Y_pred5)


# In[33]:


AC = accuracy_score(Y_test, Y_pred1) * 100
AC


# In[34]:


from sklearn.metrics import classification_report
print("Classification Report of Logistic Regression: \n", classification_report(Y_test, Y_pred1, digits = 4))


# In[35]:


filename = "pred.sav"
pickle.dump(lr, open(filename, "wb"))


# In[36]:


loaded_model = pickle.load(open("pred.sav", "rb"))


# In[37]:


input_data = (5,166,72,19,175,25.8,0.587,51)
A = np.asarray(input_data)
B = A.reshape(1, -1)
prediction = loaded_model.predict(B)
print("prediction")
if(prediction[0]==0):
    print("You are Non-Diabetic")
else:
    print("You are Diabetic")

