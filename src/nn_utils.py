# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:08:05 2019

@author: cmy
"""
import h5py

def cmp(t1,t2):
    '''比较两个tuple是否相等'''
    if len(t1) != len(t2):
        return 1
    t1 = set(t1)
    t2 = set(t2)
    if len(t1) == len(t1.intersection(t2)) and len(t2)==len(t1.intersection(t2)):
        return 0
    else:
        return 1
    
def compute_acc(list1,list2):
    '''计算查询路径的准确率'''
    num = len(list1)
    conum = 0
    for i in range(len(list1)):
        if cmp(list1[i],list2[i]) == 0:
            conum+=1
    return float(conum)/float(num)

def GenerateSortedTuples(candidate_tuples_valid,predict_score,question2sample):
    predict_tuples = []
    #print (len(candidate_tuples_valid),len(predict_score),len(question2sample))
    for i in range(len(question2sample)):
        if len(question2sample[i])==0:
            predict_tuples.append([])
            continue
        begin = question2sample[i][0]
        end = question2sample[i][-1]+1
        score = [s for s in predict_score[begin:end]]
        tuples = [t for t in candidate_tuples_valid[begin:end]]
        sample_prop = [each for each in zip(score,tuples)]#(prop,(tuple))
        sample_prop = sorted(sample_prop, key=lambda x:x[0], reverse=True)
        sorted_tuples = [t[1] for t in sample_prop]
        predict_tuples.append(sorted_tuples)
    assert len(question2sample) == len(predict_tuples)
    return predict_tuples

def compute_recall(predict_tuples_valid,gold_tuples_valid,topn):
    '''
    根据文本匹配结果得到候选实体顺序，提取topn个候选答案
    在整个语料集上统计recall
    '''
    assert len(predict_tuples_valid) == len(gold_tuples_valid)
    true_num = 0
    hop1_num = 0
    hop1_true_num = 0
    for i in range(len(predict_tuples_valid)):
        if len(predict_tuples_valid[i])>topn:
            tuples = predict_tuples_valid[i][:topn]
        else:
            tuples = predict_tuples_valid[i]
        if len(gold_tuples_valid[i])<=3:
            hop1_num += 1
        for t in tuples:
            if cmp(t,gold_tuples_valid[i]) == 0:
                true_num += 1
                if len(gold_tuples_valid[i])<=3:
                    hop1_true_num += 1
                break
    return float(true_num)/len(gold_tuples_valid),float(hop1_true_num)/hop1_num

def compute_recall_subject(predict_top_entity,gold_entitys_valid,topn):
    '''
    根据文本匹配结果得到候选实体顺序，提取topn个候选实体，并与标注实体计算recall
    只考虑单实体问题
    '''
    assert len(predict_top_entity) == len(gold_entitys_valid)
    true_num = 0
    hop1_num = 0
    for i in range(len(predict_top_entity)):
        if len(gold_entitys_valid[i]) == 1:
            hop1_num += 1
            if len(predict_top_entity[i])>topn:
                entitys = predict_top_entity[i][:topn]
            else:
                entitys = predict_top_entity[i]
            for e in entitys:
                if e == gold_entitys_valid[i][0]:
                    true_num += 1
                    break
    return float(true_num)/hop1_num
def GenerateTopEntity(candidate_tuples_valid,predict_score,question2sample,topk=30):
    '''对每个问题，对候选答案排序，选取前topk个候选关系，进一步对候选实体排序'''
    predict_subjects = []
    for i in range(len(question2sample)):
        if len(question2sample[i])==0:
            predict_subjects.append([])
            continue
        begin = question2sample[i][0]
        end = question2sample[i][-1]+1
        score = [s for s in predict_score[begin:end]]
        tuples = [t for t in candidate_tuples_valid[begin:end]]
        sample_prop = [each for each in zip(score,tuples)]#(prop,(tuple))
        sample_prop = sorted(sample_prop, key=lambda x:x[0], reverse=True)
        if len(sample_prop)>topk:
            sample_prop = sample_prop[:topk]
        sorted_tuples = [t[1] for t in sample_prop]
        subject2num = {}
        for t in sorted_tuples:
            if t[0] in subject2num:
                subject2num[t[0]] += 1
            else:
                subject2num[t[0]] = 1
        subject_prop = [[subject2num[s],s] for s in subject2num]
        subject_prop = sorted(subject_prop, key=lambda x:x[0], reverse=True)
        sorted_subjects = [s[1] for s in subject_prop]
        predict_subjects.append(sorted_subjects)
    return predict_subjects

def computeFscore(gold_ans,pre_ans):
    '''计算答案的F值'''
    gold_num = 0.0
    pre_num = 0.0
    true_num = 0.0
    for i in range(len(gold_ans)):
        gold_num += len(gold_ans[i])
        pre_num += len(pre_ans[i])
        true_num += len(set(gold_ans[i]).intersection(set(pre_ans[i])))
    try:
        p = true_num/pre_num
        r = true_num/gold_num
        f = 2*p*r/(p+r)
        return p,r,f
    except:
        return 0.0,0.0,0.0
    
def save_model(address,model):#保存keras神经网络模型
    f = h5py.File(address,'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        f.create_dataset('weight' + str(i),data = weight[i])
    f.close()
def load_model(address, model):#下载keras神经网络模型
    f = h5py.File(address, 'r')
    weight = []
    for i in range(len(f.keys())):
        weight.append(f['weight' + str(i)][:])
    model.set_weights(weight)