#coding=utf-8
import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from nltk.tokenize import word_tokenize


"""
'I'm super man'
tokenize:
['I', ''m', 'super','man' ]
"""

df= pd.read_csv('traindata.csv')
df['Text']=df['text']
df['Label']= df['label']
data= df.drop(columns=['text','label'])

dftest= pd.read_csv('testdata.csv')
dftest['Text']=dftest['text']
dftest['Label']= dftest['label']
dftest= dftest.drop(columns=['text','label'])

##%% import google word2vec
#from gensim.models import KeyedVectors
#word2vec_path = '/Users/apple/Desktop/SEM 8/CS680_nlp/Project/embeddings/GoogleNews-vectors-negative300.bin'
#word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

#%%
import re
import string
def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct
data['Text_Clean'] = data['Text'].apply(lambda x: remove_punct(x))
dftest['Text_Clean'] = dftest['Text'].apply(lambda x: remove_punct(x))

tokens = [word_tokenize(sen) for sen in data.Text_Clean]
tokenstst = [word_tokenize(sen) for sen in dftest.Text_Clean]

def lower_token(tokens): 
    return [w.lower() for w in tokens]    
    
lower_tokens = [lower_token(token) for token in tokens]
lower_tokenstst = [lower_token(token) for token in tokenstst]

from nltk.corpus import stopwords
stoplist = stopwords.words('english')
def removeStopWords(tokens): 
    return [word for word in tokens if word not in stoplist]
filtered_words = [removeStopWords(sen) for sen in lower_tokens]
filtered_wordstst = [removeStopWords(sen) for sen in lower_tokenstst]
data['Text_Final'] = [' '.join(sen) for sen in filtered_words]
data['tokens'] = filtered_words

dftest['Text_Final'] = [' '.join(sen) for sen in filtered_wordstst]
dftest['tokens'] = filtered_wordstst

#%%
pos = []
neg = []
neut= []
for l in data.Label:
    if l == -1:
        pos.append(0)
        neg.append(1)
        neut.append(0)
    elif l == 0:
        pos.append(0)
        neg.append(0)
        neut.append(1)
    else:
        pos.append(1)
        neg.append(0)
        neut.append(0)
        
data['Pos']= pos
data['Neg']= neg
data['Neut']= neut

data = data[['Text_Final', 'tokens', 'Pos', 'Neg','Neut']]

#%%
from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data,test_size=0.2,random_state=42)

#%%
data_test=data_test.reset_index()
data_train=data_train.reset_index()

#%%
training_sequences= []
for i in range(len(data_train)):
    temp= np.zeros(300)
    l= len(data_train['tokens'][i])
    for j in range(len(data_train['tokens'][i])):
        word= data_train['tokens'][i][j]
        if word not in word2vec.vocab:
            word= 'ukn'
        vector= word2vec.word_vec(word)
        temp=temp+vector/l
    training_sequences.append(temp)
    
test_sequences= []
for i in range(len(data_test)):
    temp= np.zeros(300)
    l= len(data_test['tokens'][i])
    for j in range(len(data_test['tokens'][i])):
        word= data_test['tokens'][i][j]
        if word not in word2vec.vocab:
            word= 'ukn'
        vector= word2vec.word_vec(word)
        temp=temp+vector/l
    test_sequences.append(temp)
    
all_sequences= []
for i in range(len(data)):
    temp= np.zeros(300)
    l= len(data['tokens'][i])
    for j in range(len(data['tokens'][i])):
        word= data['tokens'][i][j]
        if word not in word2vec.vocab:
            word= 'ukn'
        vector= word2vec.word_vec(word)
        temp=temp+vector/l
    all_sequences.append(temp)
    
tst_sequences= []
for i in range(len(dftest)):
    temp= np.zeros(300)
    l= len(dftest['tokens'][i])
    for j in range(len(dftest['tokens'][i])):
        word= dftest['tokens'][i][j]
        if word not in word2vec.vocab:
            word= 'ukn'
        vector= word2vec.word_vec(word)
        temp=temp+vector/l
    tst_sequences.append(temp)

train_cnn_data= training_sequences
test_cnn_data= test_sequences
tst_cnn_data= tst_sequences
all_cnn_data= all_sequences

#%%
train_dataset=[]
for i in range(len(data_train)):
    train_dataset.append([train_cnn_data[i],list([data_train["Neg"][i],data_train["Neut"][i],data_train["Pos"][i]])])
train_dataset=np.array(train_dataset)

test_dataset=[]
for i in range(len(data_test)):
    test_dataset.append([test_cnn_data[i],list([data_test["Neg"][i],data_test["Neut"][i],data_test["Pos"][i]])])
test_dataset=np.array(test_dataset)

all_dataset=[]
for i in range(len(data)):
    all_dataset.append([all_cnn_data[i],list([0,0,0])])
all_dataset=np.array(all_dataset)

tst_dataset=[]
for i in range(len(dftest)):
    tst_dataset.append([tst_cnn_data[i],list([0,0,0])])
tst_dataset=np.array(tst_dataset)

n_input_layer = 300  # 输入层
print("n_input_layer",n_input_layer)

#%%

n_layer_1 = 1000
n_layer_2 = 1000
n_layer_3 = 800
n_layer_4 = 600
n_layer_5 = 400
n_layer_6 = 200

n_output_layer = 3

def neural_network(data):
    
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    layer_3_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_layer_3])),
                   'b_': tf.Variable(tf.random_normal([n_layer_3]))}
    layer_4_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_3, n_layer_4])),
                   'b_': tf.Variable(tf.random_normal([n_layer_4]))}
    layer_5_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_4, n_layer_5])),
                   'b_': tf.Variable(tf.random_normal([n_layer_5]))}
    layer_6_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_5, n_layer_6])),
                   'b_': tf.Variable(tf.random_normal([n_layer_6]))}
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_6, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['b_'])
    layer_3 = tf.nn.relu(layer_3)  # 激活函数
    layer_4 = tf.add(tf.matmul(layer_3, layer_4_w_b['w_']), layer_4_w_b['b_'])
    layer_4 = tf.nn.relu(layer_4)  # 激活函数
    layer_5 = tf.add(tf.matmul(layer_4, layer_5_w_b['w_']), layer_5_w_b['b_'])
    layer_5 = tf.nn.relu(layer_5)  # 激活函数
    layer_6 = tf.add(tf.matmul(layer_5, layer_6_w_b['w_']), layer_6_w_b['b_'])
    layer_6 = tf.nn.relu(layer_6)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_6, layer_output_w_b['w_']), layer_output_w_b['b_'])
    
    return layer_output

batch_size = 50

X = tf.placeholder('float', [None, len(train_dataset[0][0])])
Y = tf.placeholder('float')
#为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：

#计算操作单元用于实现反向传播算法和AdamOptimizer。然后，它返回给你的只是一个单一的操 作，
#当运行这个操作时，它用AdamOptimizer训练模型，微调变量，不断减少成本。
def train_neural_network(X, Y):
    predict = neural_network(X)#得到预测结果（通过神经网络）
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels= Y))#得到损失函数
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  #使用最优化算法来使损失函数值最小
    
    epochs = 10  #32次整体迭代
    #在一个Session里面启动模型，并且初始化变量
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        i = 0
        train_x = train_dataset[:, 0]
        train_y = train_dataset[:, 1]
        test_x = test_dataset[:, 0]
        all_x= all_dataset[:, 0]
        tst_x= tst_dataset[:, 0]
        
        for epoch in range(epochs):
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                _, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
                epoch_loss += c
                i += batch_size
            
            print('epoch:',epoch)
            i=0
            epoch_loss=0
        pred_y= session.run(predict, {X :list(test_x)})
        all_y_txt= session.run(predict, {X :list(all_x)})
        tst_y_txt= session.run(predict, {X :list(tst_x)})

    return pred_y,all_y_txt,tst_y_txt
pred_y,all_y_txt,tst_y_txt= train_neural_network(X, Y)

ytrue= []
for i in range(len(test_dataset)):
    ytrue.append(np.argmax(test_dataset[:, 1][i])-1)

ypred= np.argmax(pred_y,axis=1)-1 
    
#%%
from sklearn.metrics import classification_report
target_names = ['class -1','class 0','class 1']
print(classification_report(ytrue, ypred,target_names=target_names))

#%%
all_txt= pd.DataFrame(all_y_txt,columns=['text-1','text0','text1'])
tst_txt= pd.DataFrame(tst_y_txt,columns=['text-1','text0','text1'])

all_txt.to_csv('all_distribution_text.csv',index= None)
tst_txt.to_csv('test_distribution_text.csv',index= None)

#%%
ann_input= pd.DataFrame(all_cnn_data)
ann_input.to_csv('ann_input.csv',index= None)
