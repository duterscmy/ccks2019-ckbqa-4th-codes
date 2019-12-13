# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:20:27 2019

@author: cmy
"""

import pickle
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import numpy as np
'''该模块目的是从所有候选实体中筛选出topn个候选实体,并保证正确实体在其中'''
def GetData(corpus,mode):
    '''从语料中提取训练和预测需要的数据
    X : numpy.array, (num_sample,num_feature)
    Y : numpy.array, (num_sample,1)
    samples : python-list,(num_sample,)
    gold_tuple : python-list, (num_question,1)
    question2sample : python-dict, key:questionindex , value:sampleindexs
    mode : python-str
    '''
    X = []
    Y = []
    samples = []
    gold_entitys = []
    question2sample = {}
    sample_index = 0
    
    true_num = 0
    one_num = 0
    one_true_num = 0
    for i in range(len(corpus)):
        candidate_entitys = corpus[i]['candidate_entity']
        gold_entity = corpus[i]['gold_entitys']
        
        candidate_entitys_list = [each for each in candidate_entitys]
        if len(gold_entity) == len(set(gold_entity).intersection(set(candidate_entitys_list))):
            true_num += 1
            if len(gold_entity) == 1:
                one_true_num += 1
        if len(gold_entity) == 1:
            one_num += 1

        q_sample_indexs = []
        for e in candidate_entitys:
            features = candidate_entitys[e]
            X.append(features[1:])  # 第0个特征是该实体对应的mention 
            if e in gold_entity:
                Y.append([1])
            else:
                Y.append([0])
            samples.append(e)
            q_sample_indexs.append(sample_index)
            sample_index+=1
        gold_entitys.append(gold_entity)
        question2sample[i] = q_sample_indexs  # 每个问题i对应的sample index
    print ('所有问题候选主语召回率为：%.3f 其中单主语问题为：%.3f'%(true_num/len(corpus),one_true_num/one_num))
    X = np.array(X,dtype='float32')
    Y = np.array(Y,dtype='float32')
    return X,Y,samples,gold_entitys,question2sample



def GetPredictEntitys(prepro,samples,question2sample,topn):
    '''
    得到问题对应的样本，对它们按照概率进行排序，选取topn作为筛选后的候选实体
    对于属性值，只保留排名前3位的
    input:
        prepro : python-list [[0-prob,1-prob]]
        samples : python-list [str]
        question2sample : dict
        topn : int
    output:
        predict_entitys : list [[str]]
    '''
    predict_entitys = []
    for i in range(len(question2sample)):
        sample_indexs = question2sample[i]
        if len(sample_indexs) == 0:
            predict_entitys.append([])
            continue
        begin_index = sample_indexs[0]
        end_index = sample_indexs[-1]
        now_samples = [samples[j] for j in range(begin_index,end_index+1)]
        now_props = [prepro[j][1] for j in range(begin_index,end_index+1)]

        sample_prop = [each for each in zip(now_props,now_samples)]#(prop,(tuple))
        sample_prop = sorted(sample_prop, key=lambda x:x[0], reverse=True)
        entitys = [each[1] for each in sample_prop]
        if len(entitys)>topn:
            predict_entitys.append(entitys[:topn])
        else:
            predict_entitys.append(entitys)
            # print (predict_entitys)
    return predict_entitys

def ComputePrecision(gold_entitys,predict_entitys):
    '''
    判断每个问题预测的实体和真实的实体是否完全一致，返回正确率
    '''
    true_num = 0
    one_num = 0
    one_true_num = 0
    wrong_list = []#所有筛选实体错误的问题的序号
    for i in range(len(gold_entitys)):
        if len(set(gold_entitys[i]).intersection(set(predict_entitys[i]))) == len(gold_entitys[i]):
            true_num +=1
            if len(gold_entitys[i]) == 1:
                one_true_num +=1
        else:
            if len(gold_entitys[i]) == 1:
                wrong_list.append(i)
        if len(gold_entitys[i])==1:
            one_num+=1
            
    return float(one_true_num)/one_num,true_num/len(gold_entitys),wrong_list

def SaveFilterCandiE(corpus,predict_entitys):
    for i in range(len(corpus)):
        candidate_entity_filter = {}
        for e in predict_entitys[i]:
            # print (corpus[i]['candidate_entity'][e])
            candidate_entity_filter[e] = corpus[i]['candidate_entity'][e]
        corpus[i]['candidate_entity_filter'] = candidate_entity_filter
    return corpus

if __name__ == '__main__':
    valid_corpus = pickle.load(open('../data/candidate_entitys_valid.pkl','rb'))
    train_corpus = pickle.load(open('../data/candidate_entitys_train.pkl','rb'))
    
    #(numsample,feature),(numsample,1),(numsample,)
    x_train,y_train,samples_train,gold_entitys_train,question2sample_train = GetData(train_corpus,'train')
    x_valid,y_valid,samples_valid,gold_entitys_valid,question2sample_valid = GetData(valid_corpus,'valid')
    print(x_train.shape)
    #逻辑回归
    model = linear_model.LogisticRegression(C=1e5)
    model.fit(x_train, y_train)
    pickle.dump(model,open('../data/model/entity_classifer_model.pkl','wb'))
    y_predict = model.predict_proba(x_valid).tolist()
    
    topns = [1,2,3,5,6,7,8,9,10,15,20]
    #topns = [5]
    #得到候选实体
    for topn in topns:
        predict_entitys= GetPredictEntitys(y_predict,samples_valid,question2sample_valid,topn)
        #判断候选实体的准确性，只要有一个在真正实体中即可
        precision_topn_one,precision_topn_all,wrong_list= ComputePrecision(gold_entitys_valid,predict_entitys) 
        print ('在验证集上逻辑回归top%d筛选后，所有问题实体召回率为%.3f，单实体问题实体召回率%.3f'%(topn,precision_topn_all,precision_topn_one))
    #将筛选后的候选实体写入corpus并保存
    valid_corpus = SaveFilterCandiE(valid_corpus,predict_entitys)
    
    y_predict = model.predict_proba(x_train).tolist()
    predict_entitys = GetPredictEntitys(y_predict,samples_train,question2sample_train,topn)
    train_corpus = SaveFilterCandiE(train_corpus,predict_entitys)
    
    pickle.dump(valid_corpus,open('../data/candidate_entitys_filter_valid.pkl','wb'))
    pickle.dump(train_corpus,open('../data/candidate_entitys_filter_train.pkl','wb'))
    
