# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:47:19 2024

@author: Acer
"""

import pandas as pd
df = pd.read_csv("C:/14-Ensemble_Learning/diabetes.csv")
df.head()
df.isnull().sum()
df.describe()
df.Outcome.value_counts()


#There is slight imbalance in our dataset but since
#it is not major we will not worry about it!
#Train test split

X=df.drop("Outcome",axis="columns")
y=df.Outcome
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:3]
#In order to make your data balanced while splitting, you can 
#use stratify
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,stratify=y, random_state=10)
X_train.shape
X_test.shape
y_train.value_counts()

# 0    375
# 1    201
67/125
#Train using stand alone model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#Here k fold cross validation is used 
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
scores
scores.mean()
#Accuracy=0.7097529921059333

#train using bagging
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
    )
bag_model.fit(X_train, y_train)
bag_model.oob_score_

#Note here we are not using test data, using
#OOB samples results are tested
bag_model.score(X_test, y_test)

#Now let us apply cross validation
bag_model = BaggingClassifier(
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
    )
scores = cross_val_score(bag_model, X, y, cv=5)
scores
scores.mean()

#We can see some improvement in test scores with bagging 
#classifiers as 

#Train using random forest
from sklearn.ensemble import RandomForestClassifier
scores = cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv=5)
scores.mean()





