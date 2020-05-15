#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:25:27 2020

@author: apple
"""
#%%
import pandas as pd
import numpy as np
from PIL import Image

path= '/Users/apple/Desktop/SEM 8/CS680_nlp/Project/data/Raw_traindata.csv'
df= pd.read_csv(path)
df['text_corrected'][119]=df['text_ocr'][119]

df['text']=df['text_corrected']
for i in range(len(df)):
    if((pd.isna(df['text_corrected'][i]))):
        print(i)
        df['text'][i]='unknown'

label= np.zeros(len(df))
label=label.astype('int')

pos= 0
neg=0
neut=0
for i in range(len(df)):
    if((df['overall_sentiment'][i]=='very_negative')or(df['overall_sentiment'][i]=='negative')):
        label[i]= -1
        neg=neg+1
    elif((df['overall_sentiment'][i]=='very_positive')or(df['overall_sentiment'][i]=='positive')):
        label[i]= 1
        pos=pos+1
    else:
        label[i]= 0
        neut=neut+1
        
df['label']= label

df= df.drop(columns=['Unnamed: 0','text_ocr','text_corrected','humour','sarcasm','offensive','motivational','overall_sentiment'])
print('pos:',pos)
print('neut:',neut)
print('neg:',neg)

#%%
from sklearn.utils import shuffle
df= shuffle(df)
df=df.reset_index()
df= df.drop(columns=['index'])

frac=0.20
df.iloc[:int((1-frac)*len(df)),:].to_csv('traindata.csv',index=None)
df.iloc[int((1-frac)*len(df)):,:].to_csv('testdata.csv',index=None)

#%%
path= '/Users/apple/Desktop/SEM 8/CS680_nlp/Project/data/train_images/'
df= pd.read_csv('traindata.csv')
for i in range(len(df)):
    if(df['label'][i]==-1):
        img= Image.open(path+df['image_name'][i])
        img.save('neg/'+df['image_name'][i])
    elif(df['label'][i]==0):
        img= Image.open(path+df['image_name'][i])
        img.save('neut/'+df['image_name'][i])
    else:
        img= Image.open(path+df['image_name'][i])
        img.save('pos/'+df['image_name'][i])
