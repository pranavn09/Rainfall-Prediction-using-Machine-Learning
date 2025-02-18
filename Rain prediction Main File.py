#!/usr/bin/env python
# coding: utf-8

# # RAINFALL PREDICTION

# In[3]:


from IPython import display
display.Image(r"C:\Users\Pranav\OneDrive\Desktop\GIF.gif")


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from catboost import CatBoostClassifier
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv("weatherAUS.csv")
df.head(10)


# # Data Exploration

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


numerical_feature = [feature for feature in df.columns if df[feature].dtypes != 'O']
discrete_feature=[feature for feature in numerical_feature if len(df[feature].unique())<25]
continuous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]
categorical_feature = [feature for feature in df.columns if feature not in numerical_feature]
print("Numerical Features Count {}".format(len(numerical_feature)))
print("Discrete feature Count {}".format(len(discrete_feature)))
print("Continuous feature Count {}".format(len(continuous_feature)))
print("Categorical feature Count {}".format(len(categorical_feature)))


# In[9]:


df.isnull().sum()*100/len(df)


# In[10]:


#pip install pandas-profiling


# In[11]:


#from pandas_profiling import ProfileReport
#prof = ProfileReport(df)
#prof.to_file(output_file='report.html')


# In[12]:


sns.heatmap(df.isnull(), cbar=False, cmap='PuBu')


# In[13]:


print(numerical_feature)


# In[14]:


def randomsampleimputation(df, variable):
    df[variable]=df[variable]
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable]=random_sample


# In[15]:


randomsampleimputation(df, "Cloud9am")
randomsampleimputation(df, "Cloud3pm")
randomsampleimputation(df, "Evaporation")
randomsampleimputation(df, "Sunshine")


# In[16]:


df.head(10)


# In[17]:


print(continuous_feature)


# In[18]:


for feature in continuous_feature:
    data=df.copy()
    sns.distplot(df[feature])
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.figure(figsize=(15,15))
    plt.show()


# In[19]:


for feature in continuous_feature:
    data=df.copy()
    sns.boxplot(data[feature])
    plt.title(feature)
    plt.figure(figsize=(15,15))


# In[20]:


for feature in continuous_feature:
    if(df[feature].isnull().sum()*100/len(df))>0:
        df[feature] = df[feature].fillna(df[feature].median())


# In[21]:


df.isnull().sum()*100/len(df)


# In[22]:


discrete_feature


# In[23]:


def mode_nan(df,variable):
    mode=df[variable].value_counts().index[0]
    df[variable].fillna(mode,inplace=True)
mode_nan(df,"Cloud9am")
mode_nan(df,"Cloud3pm")


# In[24]:


df["RainToday"] = pd.get_dummies(df["RainToday"], drop_first = True)
df["RainTomorrow"] = pd.get_dummies(df["RainTomorrow"], drop_first = True)


# In[25]:


encoder = preprocessing.LabelEncoder() 
df['Location']= encoder.fit_transform(df['Location'])
df['WindGustDir']= encoder.fit_transform(df['WindGustDir']) 
df['WindDir9am']= encoder.fit_transform(df['WindDir9am']) 
df['WindDir3pm']= encoder.fit_transform(df['WindDir3pm'])


# In[26]:


df["WindGustDir"] = df["WindGustDir"].fillna(df["WindGustDir"].value_counts().index[0])
df["WindDir9am"] = df["WindDir9am"].fillna(df["WindDir9am"].value_counts().index[0])
df["WindDir3pm"] = df["WindDir3pm"].fillna(df["WindDir3pm"].value_counts().index[0])


# In[27]:


df.isnull().sum()*100/len(df)


# In[28]:


df.head(10)


# In[29]:


df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%dT", errors = "coerce")


# In[30]:


df["Date_year"] = df["Date"].dt.year
df["Date_month"] = df["Date"].dt.month
df["Date_day"] = df["Date"].dt.day
df


# In[31]:


corrmat = df.corr()
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(corrmat,annot=True)


# In[32]:


def qq_plots(df, variable):
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.show()


# In[33]:


for feature in continuous_feature:
    print(feature)
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[feature].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[feature], dist="norm", plot=plt)
    plt.show()


# In[34]:


#df.to_csv("preprocessed.csv", index=False)


# In[35]:


X = df.drop(["RainTomorrow", "Date"], axis=1)
Y = df["RainTomorrow"]


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size =0.25, random_state = 24)


# In[61]:


X_train


# In[62]:


y_train


# In[63]:


sm=SMOTE(random_state=24)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_res)))


# In[64]:


#!pip install catboost


# In[65]:


cat = CatBoostClassifier(iterations=2000, eval_metric = "AUC")
cat.fit(X_train_res, y_train_res)


# In[66]:


from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[67]:


y_pred = cat.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[68]:


metrics.plot_roc_curve(cat, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred, average=None) 


# In[69]:


from sklearn.ensemble import RandomForestClassifier


# In[70]:


rf=RandomForestClassifier()
rf.fit(X_train_res,y_train_res)


# In[71]:


y_pred1 = rf.predict(X_test)
print(confusion_matrix(y_test,y_pred1))
print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))


# In[72]:


metrics.plot_roc_curve(rf, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred1, average=None) 


# In[73]:


from sklearn.linear_model import LogisticRegression


# In[74]:


logreg = LogisticRegression()
logreg.fit(X_train_res, y_train_res)


# In[75]:


y_pred2 = logreg.predict(X_test)
print(confusion_matrix(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))
print(classification_report(y_test,y_pred2))


# In[76]:


metrics.plot_roc_curve(logreg, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred2, average=None) 


# In[77]:


#pip install xgboost


# In[78]:


from xgboost import XGBClassifier


# In[79]:


xgb = XGBClassifier()
xgb.fit(X_train_res, y_train_res)


# In[80]:


y_pred3 = xgb.predict(X_test)
print(confusion_matrix(y_test,y_pred3))
print(accuracy_score(y_test,y_pred3))
print(classification_report(y_test,y_pred3))


# In[81]:


metrics.plot_roc_curve(xgb, X_test, y_test)
metrics.roc_auc_score(y_test, y_pred3, average=None) 


# In[82]:


import joblib


# In[83]:


joblib.dump(rf, "rf.pkl")
joblib.dump(cat, "cat.pkl")
joblib.dump(logreg, "logreg.pkl")
joblib.dump(xgb, "xgb.pkl")


# In[ ]:




