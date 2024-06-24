#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("netflix_titles.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe(include="all")


# In[7]:


data.isnull().sum()


# In[8]:


data["date_added"]=pd.to_datetime(data["date_added"],errors="coerce")


# In[9]:


data["director"]=data["director"].fillna("No_director")
data["cast"]=data["cast"].fillna("NO_cast")
data["country"]=data["country"].fillna(data["country"].mode()[0])


# In[10]:


data


# In[11]:


data["date_added"]=data["date_added"].fillna(data["date_added"].mode()[0])


# In[12]:


data.dropna(inplace=True)


# In[13]:


data.isnull().sum()


# # EDA

# In[14]:


sns.countplot(x="type",data=data)
plt.title("Type of content types")


# In[15]:


Top_countries =data["country"].value_counts().head(10)
sns.barplot(x=Top_countries.values,y=Top_countries.index)
plt.title("Top countries")


# In[16]:


data["country"].value_counts().head(25)


# In[17]:


plt.figure(figsize=(14, 7))
sns.countplot(x='release_year', data=data, order=data['release_year'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Trend of Content Release Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.show()


# In[18]:


Top_ratings = data["rating"].value_counts().head(19)
sns.barplot(x=Top_ratings.values,y=Top_ratings.index)
plt.title("Top ratings")


# In[19]:


data["director"].value_counts().head(15)


# In[20]:


data.head()


# In[21]:


titles_by_type_country = data.groupby(['type', 'country']).size().unstack().fillna(0)
titles_by_type_country


# # Model development

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,classification_report


# In[23]:


label_encoders=[]
for column in ["type","director","cast","country","rating","duration","listed_in"]:
    le = LabelEncoder()
    data[column]=le.fit_transform(data[column])
    #label_encoders[column] =le


# In[24]:


le


# In[25]:


x=data[["director","cast","country","rating","duration","listed_in"]]
y=data["type"]


# # feature_importances_

# In[41]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(x,y)
print(gb.feature_importances_)


# In[42]:


vis = pd.Series(gb.feature_importances_,index=x.columns)
vis.nlargest(20).plot(kind='barh')
plt.title("feature_importances")
plt.show()


# In[26]:


x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=42)


# # Scaling

# In[27]:


ss=StandardScaler()
x_train_sc=ss.fit_transform(x_train)
x_test_sc = ss.transform(x_test)


# # logistic regression 

# In[28]:


lr =LogisticRegression()
lr.fit(x_train,y_train)
y_pre =lr.predict(x_test)
print("accuracy_score:",accuracy_score(y_test,y_pre))


# In[29]:


lr =LogisticRegression(random_state=42)
lr.fit(x_train_sc,y_train)
y_pre_sc =lr.predict(x_test_sc)
print("accuracy_score:",accuracy_score(y_test,y_pre_sc))


# In[30]:


print(classification_report(y_test,y_pre))


# # Random forest

# In[31]:


Rc=RandomForestClassifier(random_state=42)
Rc.fit(x_train,y_train)
y_pred_rc = Rc.predict(x_test)
print("random_accuracy_score:",accuracy_score(y_test,y_pred_rc))


# In[32]:


Rc=RandomForestClassifier(random_state=42)
Rc.fit(x_train_sc,y_train)
y_pred_rc_sc = Rc.predict(x_test_sc)
print("random_accuracy_score:",accuracy_score(y_test,y_pred_rc_sc))


# In[33]:


print(classification_report(y_test,y_pred_rc))


# # SVC

# In[34]:


sc=SVC()
sc.fit(x_train,y_train)
y_pred_sc = sc.predict(x_test)
print("random_accuracy_score:",accuracy_score(y_test,y_pred_sc))


# In[35]:


sc=SVC()
sc.fit(x_train_sc,y_train)
y_pred_ss = sc.predict(x_test_sc)
print("random_accuracy_score:",accuracy_score(y_test,y_pred_ss))


# In[36]:


print(classification_report(y_test,y_pred_sc))


# # Model selection 

# In[37]:


print("accuracy_score:",accuracy_score(y_test,y_pre))
print("random_accuracy_score:",accuracy_score(y_test,y_pred_rc))
print("svm_accuracy_score:",accuracy_score(y_test,y_pred_sc))


# In[45]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [0, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[47]:


grid_search = GridSearchCV(estimator=Rc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)


# In[49]:


grid_search.fit(x_train, y_train)


# In[50]:


best_params = grid_search.best_params_
print(f'Best parameters for Random Forest: {best_params}')


# In[ ]:




