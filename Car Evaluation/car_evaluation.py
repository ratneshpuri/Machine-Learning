#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


cars=pd.read_csv("car_evaluation.csv")


# In[3]:


cars.head()


# In[4]:


cars.columns=['Buying_Price','Maintainence_Cost','No_of_Doors','Persons_Capacity','lug_boot','Safety','Acceptibility']


# In[5]:


cars.head()


# In[6]:


cars.shape


# In[7]:


cars.info()


# In[8]:


cars.isnull().sum()


# In[9]:


for i in cars.columns:
    print(cars[i].value_counts())


# In[10]:


plt.figure(figsize=(10,10))
sns.barplot(cars.Acceptibility,cars.index)
plt.xlabel("Acceptibility")
plt.ylabel("No. of Attributes")
plt.show()


# In[11]:


plt.figure(figsize=(10,10))
sns.barplot(cars.Buying_Price,cars.index)
plt.xlabel("Buying_Price")
plt.ylabel("No. of Attributes")
plt.show()


# In[12]:


plt.figure(figsize=(10,10))
sns.barplot(cars.Maintainence_Cost,cars.index)
plt.xlabel("Maintainence_Cost")
plt.ylabel("No. of Attributes")
plt.show()


# In[13]:


plt.figure(figsize=(10,10))
sns.barplot(cars.No_of_Doors,cars.index)
plt.xlabel("No. of Doors")
plt.ylabel("No. of Attributes")
plt.show()


# In[14]:


plt.figure(figsize=(10,10))
sns.barplot(cars.Persons_Capacity,cars.index)
plt.xlabel("Person_Capacity")
plt.ylabel("No. of Attributes")
plt.show()


# In[15]:


plt.figure(figsize=(10,10))
sns.barplot(cars.lug_boot,cars.index)
plt.xlabel("lug_boot")
plt.ylabel("No. of Attributes")
plt.show()


# In[16]:


cars.sample(5)


# In[17]:


cars.Acceptibility=cars.Acceptibility.replace(['unacc','acc','good','vgood'],[0,1,2,3])


# In[18]:


cars.No_of_Doors=cars.No_of_Doors.replace(['5more'],[5])


# In[19]:


cars.Persons_Capacity=cars.Persons_Capacity.replace(['more'],[5])


# In[20]:


cars.head()


# In[21]:


cars.Acceptibility.value_counts()


# In[22]:


cars.No_of_Doors.value_counts()


# In[23]:


cars.Persons_Capacity.value_counts()


# In[24]:


cars=cars.astype({'Persons_Capacity':'int64'})


# In[25]:


cars=cars.astype({'No_of_Doors':'int64'})


# In[26]:


cars.info()


# In[27]:


cars_dummies=pd.get_dummies(cars,drop_first=True)


# In[28]:


cars_dummies.head()


# In[29]:


x=cars_dummies.drop(['Acceptibility'],axis=1)
y=cars['Acceptibility']


# In[56]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)


# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[59]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)


# # Building Guassian Model

# In[60]:


from sklearn.naive_bayes import GaussianNB


# In[61]:


model=GaussianNB()


# In[62]:


model.fit(x_train,y_train)


# In[63]:


model.score(x_test,y_test)


# In[64]:


model.score(x_train,y_train)


# In[65]:


y_pred=model.predict(x_test)


# In[66]:


print("Accuracy score is: ",accuracy_score(y_test,y_pred))


# In[67]:


print(classification_report(y_test,y_pred))


# In[68]:


c_m=confusion_matrix(y_test,y_pred)


# In[69]:


c_m


# In[70]:


plt.figure(figsize=(10,10))
sns.heatmap(c_m,annot=True,cmap='magma')


# In[ ]:





# # Building Logistic Regression 

# In[71]:


from sklearn.linear_model import LogisticRegression


# In[72]:


model2=LogisticRegression()


# In[73]:


model2.fit(x_train,y_train)


# In[74]:


model2.score(x_test,y_test)


# In[75]:


y_pred2=model2.predict(x_test)


# In[76]:


print("Accurcy of the logistic model is: ",accuracy_score(y_test,y_pred2))


# In[77]:


print(classification_report(y_test,y_pred2))


# In[78]:


c_m2=confusion_matrix(y_test,y_pred2)
print(c_m2)


# In[79]:


plt.figure(figsize=(10,10))
sns.heatmap(c_m2,annot=True,cmap='plasma')


# In[ ]:




