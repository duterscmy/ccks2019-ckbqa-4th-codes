# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:33:42 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
import pickle
from keras_bert import load_trained_model_from_checkpoint, Tokenizer,get_custom_objects
import codecs
from keras.layers import Input,Dense,LSTM
from keras.layers.wrappers import TimeDistributed,Bidirectional
from keras.models import Model,load_model
from keras.optimizers import Adam

max_seq_len = 20
config_path = '../../news_classifer_task/wwm/bert_config.json'
checkpoint_path = '../../news_classifer_task/wwm/bert_model.ckpt'
dict_path = '../../news_classifer_task/wwm/vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
        
train_corpus = pickle.load(open('../data/corpus_train.pkl','rb'))
train_questions = [train_corpus[i]['question'] for i in range(len(train_corpus))]
train_entitys = [train_corpus[i]['gold_entitys'] for i in range(len(train_corpus))]
train_entitys = [[entity[1:-1].split('_')[0] for entity in line]for line in train_entitys]

test_corpus = pickle.load(open('../data/corpus_test.pkl','rb'))
test_questions = [test_corpus[i]['question'] for i in range(len(test_corpus))]
test_entitys = [test_corpus[i]['gold_entitys'] for i in range(len(test_corpus))]
test_entitys = [[entity[1:-1].split('_')[0] for entity in line]for line in test_entitys]
#获取输入的token_ids和label_ids
tokenizer = Tokenizer(token_dict)

def find_lcsubstr(s1, s2): 
    m=[[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0  #最长匹配的长度
    p=0 #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
            if m[i+1][j+1]>mmax:
                mmax=m[i+1][j+1]
                p=i+1
    return s1[p-mmax:p]

def GetXY(questions,entitys):
    X1, X2, Y = [], [], []
    for i in range(len(questions)):
        q = questions[i]
        x1, x2 = tokenizer.encode(first=q,max_len = max_seq_len)#分别是 词索引序列和分块索引序列
        y = [[0] for j in range(max_seq_len)]
        assert len(x1)==len(y)
        for e in entitys[i]:
            #得到实体名和问题的最长连续公共子串
            e = find_lcsubstr(e,q)
            if e in q:
                begin = q.index(e)+1
                end = begin + len(e)
                if end < max_seq_len-1:
                    for pos in range(begin,end):
                        y[pos] = [1]
        print (q)
        print (x1)
        print (y)
        X1.append(x1)
        X2.append(x2)
        Y.append(y)
    return np.array(X1),np.array(X2),np.array(Y)

trainx1,trainx2,trainy = GetXY(train_questions,train_entitys)#(num_sample,max_len)
testx1,testx2,testy = GetXY(test_questions,test_entitys)
print (trainx1.shape)

#搭建模型
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)#这里预训练的bert模型被看待为一个keras层
for l in bert_model.layers:
    l.trainable = True
x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
x = bert_model([x1_in, x2_in])#(batch,step,feature)
x = Bidirectional(LSTM(512,return_sequences=True,recurrent_dropout=0.2))(x)
p = Dense(1, activation='sigmoid')(x)
model = Model([x1_in, x2_in], p)
model.compile(loss='binary_crossentropy',optimizer=Adam(1e-5),metrics=['accuracy'])
model.summary()

#训练模型
maxf = 0.0
def computeF(gold_entity,pre_entity):
    '''
    根据标注的实体位置和预测的实体位置，计算prf,完全匹配
    输入： Python-list  3D，值为每个实体的起始位置列表[begin，end]
    输出： float
    '''    
    truenum = 0
    prenum = 0
    goldnum = 0
    for i in range(len(gold_entity)):
        goldnum += len(gold_entity[i])
        prenum  += len(pre_entity[i])
        truenum += len(set(gold_entity[i]).intersection(set(pre_entity[i])))
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall /( precise + recall)) 
    except:
        precise = recall = f = 0.0
    print('本轮实体的F值是 %f' %(f))
    return precise,recall,f

def restore_entity_from_labels_on_corpus(predicty,questions):
    def restore_entity_from_labels(labels,question):
        entitys = []
        str = ''
        labels = labels[1:-1]
        for i in range(min(len(labels),len(question))):
            if labels[i]==1:
                str += question[i]
            else:
                if len(str):
                    entitys.append(str)
                    str = ''
        if len(str):
            entitys.append(str) 
        return entitys
    all_entitys = []
    for i in range(len(predicty)):
        all_entitys.append(restore_entity_from_labels(predicty[i],questions[i]))
    return all_entitys

model = load_model('../data/model/ner_model.h5', custom_objects=get_custom_objects())
for i in range(20):
    model.fit([trainx1,trainx2],trainy, epochs=1, batch_size=64)
    predicty = model.predict([testx1,testx2],batch_size=64).tolist()#(num,len,1)
    predicty = [[1 if each[0]>0.5 else 0 for each in line] for line in predicty]
    predict_entitys = restore_entity_from_labels_on_corpus(predicty,test_questions)
    for j in range(300,320):
        print (predict_entitys[j])
        print (test_entitys[j])
    p,r,f = computeF(test_entitys,predict_entitys)
    print ('%d epoch f-score is %.3f'%(i,f))
    if f>maxf:
        model.save('../data/model/ner_model.h5')
        maxf = f
        
model = load_model('../data/model/ner_model.h5', custom_objects=get_custom_objects())
predicty = model.predict([testx1,testx2],batch_size=32).tolist()#(num,len,1)
predicty = [[1 if each[0]>0.5 else 0 for each in line] for line in predicty]
predict_entitys = restore_entity_from_labels_on_corpus(predicty,test_questions)
for j in range(300,320):
    print (predict_entitys[j])
    print (test_entitys[j])
