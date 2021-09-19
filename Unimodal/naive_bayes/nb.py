from textblob.classifiers import NaiveBayesClassifier
import nltk
import pandas as pd
import numpy as np
nltk.download('punkt')
from sklearn.metrics import classification_report
from datetime import datetime
import sys
dft=pd.read_csv('train.csv') # contains training data (text,label)
# simple upsampling by making data points of a class double, triple, etc
a=dft.loc[dft['label']==0]
b=dft.loc[dft['label']==1]
c=dft.loc[dft['label']==2]
# 3 args provided through the command line (e.g. python nb.py 2 3 1, doubles class '0', triples class '1')
for i in range(int(sys.argv[1])-1):
    dft=dft.append(a)
for i in range(int(sys.argv[2])-1):
    dft=dft.append(b)
for i in range(int(sys.argv[3])-1):
    dft=dft.append(c)
print(len(dft))
print(dft['label'].value_counts())
dft.to_csv('trainM.csv',index=False) # modified training data
#############################
now = datetime.now()
Start_Train = now.strftime("%H:%M:%S")
print("Start_Train =", Start_Train)
#############################
with open('trainM.csv', 'r') as fp:
    cl = NaiveBayesClassifier(fp, format="csv") # default Naive Bayes Classifier from TextBlob
#############################
now = datetime.now()
End_Train = now.strftime("%H:%M:%S")
print("End_Train =", End_Train)
#############################
df1=pd.read_csv('test.csv') # contains test data (id, text)
df1['label']= df1['id']    
for i in range(len(df1)):
    df1['label'][i]=cl.classify(df1['sentence'][i])
df1.to_csv('testM.csv') # modified test data containing labels (predictions) 
