# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:37:48 2019

@author: cmy
"""
import jieba
import json
import codecs as cs
import re
import time
import pickle
#relation_dic = pickle.load(open('../data/relation_dic.pkl','rb'))
def LoadCorpus(path):
    corpus = {}
    question_num = 0
    e1hop1_num = 0
    e1hop2_num = 0
    e2hop2_num = 0
    with cs.open(path,'r','utf-8') as fp:
        text = fp.read().split('\r\n\r\n')[:-1]
        for i in range(len(text)):
            #对问题进行预处理
            question = text[i].split('\r\n')[0].split(':')[1]
            question = re.sub('我想知道','',question)
            question = re.sub('你了解','',question)
            question = re.sub('请问','',question)
            
            answer = text[i].split('\n')[2].split('\t')
            sql = text[i].split('\n')[1]
            sql = re.findall('{.+}',sql)[0]
            elements = re.findall('<.+?>|\".+?\"|\?\D',sql)+re.findall('\".+?\"',sql)
            #elements中包含创引号的项目可能有重复，需要去重
            new_elements = []
            for e in elements:
                if e[0]=='\"':
                    if e not in new_elements:
                        new_elements.append(e)
                else:
                    new_elements.append(e)
            elements = new_elements
            gold_entitys = []
            gold_relations = []
            for j in range(len(elements)):
                if elements[j][0]=='<' or elements[j][0]=='\"':
                    if j%3==1:
                        gold_relations.append(elements[j])
                    else:
                        gold_entitys.append(elements[j])
            
            gold_tuple = tuple(gold_entitys+gold_relations)
            dic = {}
            dic['question'] = question#问题字符串
            dic['answer'] = answer#问题的答案
            dic['gold_tuple'] = gold_tuple
            dic['gold_entitys'] = gold_entitys
            dic['gold_relations'] = gold_relations
            dic['sql'] = sql
            corpus[i] = dic
            
            #一些统计信息
            if len(gold_entitys) == 1 and len(gold_relations) == 1:
                e1hop1_num += 1
            elif len(gold_entitys) == 1 and len(gold_relations) == 2:
                e1hop2_num += 1
            elif len(gold_entitys) == 2 and len(gold_relations) == 2:
                e2hop2_num += 1
            elif len(gold_entitys) == 2 and len(gold_relations) < 2:
                print (elements)
                print (dic['gold_entitys'])
                print (dic['sql'])
                print ('\n')
            question_num += 1
    print ('语料集问题数为%d==单实体单关系数为%d====单实体双关系数为%d==双实体双关系数为%d==总比例为%.3f\n'\
           %(question_num,e1hop1_num,e1hop2_num,e2hop2_num,(e1hop1_num+e1hop2_num+e2hop2_num)/question_num))
    return corpus

inputpaths = ['../corpus/task6ckbqa_train_2019.txt','../corpus/task6ckbqa_valid_2019.txt','../corpus/task6ckbqa_test_2019.txt']
outputpaths = ['../data/corpus_train.pkl','../data/corpus_valid.pkl','../data/corpus_test.pkl']
corpuses = []
for i in range(len(inputpaths)):
    inputpath = inputpaths[i]
    corpus = LoadCorpus(inputpath)
    outputpath = outputpaths[i]
    pickle.dump(corpus,open(outputpath,'wb'))
