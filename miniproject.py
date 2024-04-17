

# In[1]:


#import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle


# In[2]:


#loading the dataset
df_drug = pd.read_csv(r'C:\Users\Varshith\OneDrive\Desktop\mini project app\drug200.csv')


# In[3]:


#reading the first 5 rows of the dataset to examine it
df_drug.head()


# In[4]:


#checking if the dataset consists of any null values
print(df_drug.info())


# In[5]:


#exploring the categorical variables in the dataset

#counting the different drug type.
df_drug.Drug.value_counts()


# In[6]:


#counting the different sex type.
df_drug.Sex.value_counts()


# In[7]:


#counting the different Bp value.
df_drug.BP.value_counts()


# In[8]:


#counting the cholestrol valies.
df_drug.Cholesterol.value_counts()


# In[9]:


#for numerical variables

df_drug.describe()


# In[10]:


skewAge = df_drug.Age.skew(axis = 0, skipna = True)
print('Age skewness: ', skewAge)

skewNatoK = df_drug.Na_to_K.skew(axis = 0, skipna = True)
print('Na to K skewness: ', skewNatoK)


# In[11]:


sns.distplot(df_drug['Age']);


# In[12]:


#Exploratory Data Analysis

#Drug Type Distribution
sns.set_theme(style="darkgrid")
sns.countplot(y="Drug", data=df_drug, palette="flare")
plt.ylabel('Drug Type')
plt.xlabel('Total')
plt.show()


# In[13]:


#Gender Distribution

sns.set_theme(style="darkgrid")
sns.countplot(x="Sex", data=df_drug, palette="rocket")
plt.xlabel('Gender (F=Female, M=Male)')
plt.ylabel('Total')
plt.show()


# In[14]:


#Blood pressure distribution

sns.set_theme(style="darkgrid")
sns.countplot(y="BP", data=df_drug, palette="crest")
plt.ylabel('Blood Pressure')
plt.xlabel('Total')
plt.show()


# In[15]:


#choloestrol distribution

sns.set_theme(style="darkgrid")
sns.countplot(x="Cholesterol", data=df_drug, palette="magma")
plt.xlabel('Blood Pressure')
plt.ylabel('Total')
plt.show()


# In[16]:


#Gender Distribution based on Drug Type

pd.crosstab(df_drug.Sex,df_drug.Drug).plot(kind="bar",figsize=(12,5),color=['#003f5c','#ffa600','#58508d','#bc5090','#ff6361'])
plt.title('Gender distribution based on Drug type')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()


# In[17]:


#Blood Pressure Distribution based on Cholesetrol

pd.crosstab(df_drug.BP,df_drug.Cholesterol).plot(kind="bar",figsize=(15,6),color=['#6929c4','#1192e8'])
plt.title('Blood Pressure distribution based on Cholesterol')
plt.xlabel('Blood Pressure')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()


# In[18]:


#Sodium to Potassium Distribution based on Gender and Age

plt.scatter(x=df_drug.Age[df_drug.Sex=='F'], y=df_drug.Na_to_K[(df_drug.Sex=='F')], c="Blue")
plt.scatter(x=df_drug.Age[df_drug.Sex=='M'], y=df_drug.Na_to_K[(df_drug.Sex=='M')], c="Orange")
plt.legend(["Female", "Male"])
plt.xlabel("Age")
plt.ylabel("Na_to_K")
plt.show()


# In[19]:


#Data Binning
#for age
bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
df_drug['Age_binned'] = pd.cut(df_drug['Age'], bins=bin_age, labels=category_age)
df_drug = df_drug.drop(['Age'], axis = 1)


# In[20]:


#for na_to_k
bin_NatoK = [0, 9, 19, 29, 50]
category_NatoK = ['<10', '10-20', '20-30', '>30']
df_drug['Na_to_K_binned'] = pd.cut(df_drug['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
df_drug = df_drug.drop(['Na_to_K'], axis = 1)


# In[22]:


#splitting the data set into training and testing

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[23]:


X = df_drug.drop(["Drug"], axis=1)
y = df_drug["Drug"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[25]:


# feature engineering to enhance the model

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train.head()


# In[26]:


X_test.head()


# In[27]:


#To avoid overfitting we use SMOTE Techinque
# beacause the number of DRUG_Y is morethan other drugs it may overfit

from imblearn.over_sampling import SMOTE
X_train, y_train = SMOTE().fit_resample(X_train, y_train)


# In[28]:


sns.set_theme(style="darkgrid")
sns.countplot(y=y_train, data=df_drug, palette="mako_r")
plt.ylabel('Drug Type')
plt.xlabel('Total')
plt.show()


# In[29]:


#Intializing the model
#using the Logistic Regression

from sklearn.linear_model import LogisticRegression
LRclassifier = LogisticRegression(solver='liblinear', max_iter=5000)
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
LRAcc = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy is: {:.2f}%'.format(LRAcc*100))


# In[ ]:
pickle.dump(LRclassifier,open('drug.pkl','wb'))



