# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:52:22 2020

@author: Administrator
"""

"""
Created on Fri Mar 29 13:01:42 2019

@author: cmy
"""
import codecs as cs
import pickle
import time
from kb import GetRelationPaths,GetRelationPathsSingle
from similarity import BertSim
import tensorflow as tf

class TupleExtractor(object):

    def __init__(self):
        
        #加载一些缓存
        try:
            self.entity2relations_dic = pickle.load(open('../data/entity2relation_dic.pkl','rb'))
        except:
            self.entity2relations_dic = {}
            
        #加载基于tensorflow的微调过的文本匹配模型    
        self.simmer = BertSim()
        self.simmer.set_mode(tf.estimator.ModeKeys.PREDICT)
        print ('bert相似度匹配模型加载完成')
        #加载简单-复杂问题分类模型
        #self.question_classify_model = get_model()
        print ('问题分类模型加载完成')
        print ('tuples extractor loaded')
        
    def extract_tuples(self,candidate_entitys,question):
        ''''''
        candidate_tuples = {}
        entity_list = candidate_entitys.keys()#得到有序的实体列表
        inputs = []#获取所有候选路径的BERT输入
        for entity in entity_list:
            #得到该实体的所有关系路径
            starttime=time.time()
            relations = GetRelationPaths(entity)
            mention = candidate_entitys[entity][0]
            for r in relations:
                predicates = [relation[1:-1] for relation in r]#python-list 关系名列表
                human_question = '的'.join([mention]+predicates)
                inputs.append((question,human_question))
                
        #将所有路径输入BERT获得分数
        print('====共有{}个候选路径===='.format(len(inputs)))
        bert_scores = []
        batch_size = 128
        if len(inputs)%batch_size==0:
            num_batches = len(inputs)//batch_size
        else:
            num_batches = len(inputs)//batch_size + 1
        starttime=time.time()
        for i in range(num_batches):
            begin = i*batch_size
            end = min(len(inputs),(i+1)*batch_size)
            self.simmer.input_queue.put(inputs[begin:end])
            prediction = self.simmer.output_queue.get()
            bert_scores.extend([prediction[i][1] for i in range(len(prediction))])
        print ('====为所有路径计算特征耗费%.2f秒===='%(time.time()-starttime))
        
        index = 0
        for entity in entity_list:
            #得到该实体的所有关系路径
            starttime=time.time()
            relations = GetRelationPaths(entity)
            mention = candidate_entitys[entity][0]
            for r in relations:
                this_tuple = tuple([entity]+r)#生成候选tuple
                score = [entity]+candidate_entitys[entity]#初始化特征
                sim2 = bert_scores[index]
                index += 1
                score.append(sim2)
                candidate_tuples[this_tuple] = score
            print ('====得到实体%s的所有候选路径及其特征===='%(entity))

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
            print (i)
            print (question)
            candidate_tuples = self.extract_tuples(candidate_entitys,question)
            all_tuples_num += len(candidate_tuples)
            dic['candidate_tuples'] = candidate_tuples
            
            #判断gold tuple是否包含在candidate_tuples_list中
            if_true = 0
            for thistuple in candidate_tuples:
                if len(gold_tuple) == len(set(gold_tuple)&set(thistuple)):
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
        return corpus

if __name__ == '__main__':
    inputpaths = [#'../data/candidate_entitys_filter_train.pkl',
                  '../data/candidate_entitys_filter_test.pkl',
                  '../data/candidate_entitys_filter_valid.pkl']
    outputpaths = [#'../data/candidate_tuples_train.pkl',
                   '../data/candidate_tuples_test.pkl',
                   '../data/candidate_tuples_valid.pkl']
    te = TupleExtractor()
    for inputpath,outputpath in zip(inputpaths,outputpaths):
        corpus = pickle.load(open(inputpath,'rb'))
        corpus = te.GetCandidateAns(corpus)
        pickle.dump(corpus,open(outputpath,'wb'))