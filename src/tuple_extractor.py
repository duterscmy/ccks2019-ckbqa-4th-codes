# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:01:42 2019

@author: cmy
"""
import codecs as cs
import pickle
import time
from kb import GetRelationPaths
import sys
#sys.path.append("..")
#from bert4.similarity import BertSim
from similarity import BertSim
import tensorflow as tf

class TupleExtractor(object):
    def __init__(self):
        try:
            self.entity2relations_dic = pickle.load(open('../data/entity2relation_dic.pkl','rb'))
        except:
            self.entity2relations_dic = {}
        try:
            self.sentencepair2sim = pickle.load(open('../data/sentencepair2sim_dic.pkl','rb'))
        except:
            self.sentencepair2sim = {}
        self.simmer = BertSim()
        self.simmer.set_mode(tf.estimator.ModeKeys.PREDICT)
        print ('tuples extractor loaded')
        
    def extract_tuples(self,candidate_entitys,question):
        ''''''
        candidate_tuples = {}
        
        for entity in candidate_entitys:
            #得到该实体的所有关系路径
            starttime=time.time()
            
            relations = GetRelationPaths(entity)
            
            mention = candidate_entitys[entity][0]
            for r in relations:
                
                this_tuple = tuple([entity]+r)#生成候选tuple
                predicates = [relation[1:-1] for relation in r]#python-list 关系名列表

                human_question = '的'.join([mention]+predicates)
                    
                score = [entity]+[s for s in candidate_entitys[entity][0:1]]#初始化特征
                
                try:
                    sim2 = self.sentencepair2sim[question+human_question]
                except:
                    sim2 = self.simmer.predict(question,human_question)[0][1]
                    self.sentencepair2sim[question+human_question] = sim2
                self.sentencepair2sim[question+human_question] =sim2
                score.append(sim2)
                
                candidate_tuples[this_tuple] = score
            print ('====查询候选关系并计算特征耗费%.2f秒===='%(time.time()-starttime))

        return candidate_tuples
    
                
    def GetCandidateAns(self,corpus):
        '''根据mention，得到所有候选实体,进一步去知识库检索候选答案
        候选答案格式为tuple(entity,relation1,relation2) 这样便于和标准答案对比
        '''
        true_num = 0
        hop2_num = 0
        hop2_true_num = 0
        all_tuples_num = 0
        for i in range(len(corpus)):
            dic = corpus[i]
            question = dic['question']
            gold_tuple = dic['gold_tuple']
            gold_entitys = dic['gold_entitys']
            candidate_entitys = dic['candidate_entity_filter']
            
            candidate_tuples = self.extract_tuples(candidate_entitys,question)
            print (i)
            print (question)
            all_tuples_num += len(candidate_tuples)
            dic['candidate_tuples'] = candidate_tuples
            
            #判断gold tuple是否包含在candidate_tuples_list中
            if_true = 0
            for thistuple in candidate_tuples:
                if len(gold_tuple) == len(set(gold_tuple).intersection(set(thistuple))):
                    if_true = 1
                    break
            if if_true == 1:
                true_num += 1
                if len(gold_tuple) <=3 and len(gold_entitys) == 1:
                    hop2_true_num += 1
            if len(gold_tuple) <=3 and len(gold_entitys) == 1:
                hop2_num += 1
                
        print('所有问题里，候选答案能覆盖标准查询路径的比例为:%.3f'%(true_num/len(corpus)))
        print('单实体问题中，候选答案能覆盖标准查询路径的比例为:%.3f'%(hop2_true_num/hop2_num))
        print('平均每个问题的候选答案数量为:%.3f'%(all_tuples_num/len(corpus)))
        pickle.dump(self.entity2relations_dic,open('../data/entity2relation_dic.pkl','wb'))
        pickle.dump(self.sentencepair2sim,open('../data/sentencepair2sim_dic.pkl','wb'))
        return corpus

if __name__ == '__main__':
    inputpaths = ['../data/candidate_entitys_filter_valid.pkl','../data/candidate_entitys_filter_train.pkl']
    outputpaths = ['../data/candidate_tuples_valid.pkl','../data/candidate_tuples_train.pkl']
    te = TupleExtractor()
    for i in range(2):
        inputpath = inputpaths[i]
        outputpath = outputpaths[i]
        corpus = pickle.load(open(inputpath,'rb'))
        corpus = te.GetCandidateAns(corpus)
        pickle.dump(corpus,open(outputpath,'wb'))