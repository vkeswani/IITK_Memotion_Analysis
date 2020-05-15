#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:34:38 2020

@author: apple
"""
import pandas as pd
from sklearn.preprocessing import normalize

df1_txt= normalize(pd.read_csv('all_distribution_text.csv'))
df2_txt= normalize(pd.read_csv('test_distribution_text.csv'))
df1_img= normalize(pd.read_csv('all_distribution_image.csv'))
df2_img= normalize(pd.read_csv('test_distribution_image.csv'))
df1= pd.read_csv('traindata.csv')
df2= pd.read_csv('testdata.csv')

df= pd.DataFrame(df1_txt,columns=['txt-1','txt0','txt1'])
df['img-1']= df1_img[:,0]
df['img0']= df1_img[:,1]
df['img1']= df1_img[:,2]

dftest= pd.DataFrame(df2_txt,columns=['txt-1','txt0','txt1'])
dftest['img-1']= df2_img[:,0]
dftest['img0']= df2_img[:,1]
dftest['img1']= df2_img[:,2]

dfy= df1['label']
ytrue= df2['label']

X_train= df
y_train= dfy

from sklearn.utils import shuffle
X_train= shuffle(X_train)

X_train= X_train.reset_index()
X_train= X_train.drop(columns=['index'])

#%%
from sklearn.svm import SVC
clf= SVC(gamma='auto',class_weight='balanced')
clf.fit(X_train,y_train)
ypred= clf.predict(dftest)

#%%
from sklearn.metrics import classification_report
target_names = ['class -1','class 0','class 1']
print(classification_report(ytrue, ypred,target_names=target_names))
