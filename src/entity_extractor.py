# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:11:26 2019

@author: cmy
"""

import codecs as cs
import pickle
import time
from kb import GetRelations_2hop,GetRelationNum
from utils import ComputeEntityFeatures
import thulac

class SubjectExtractor(object):
    def __init__(self):
        
        self.mention2entity_dic = pickle.load(open('../data/mention2entity_dic.pkl','rb'))
        try:                
            self.entity2hop_dic = pickle.load(open('../data/entity2hop_dic.pkl','rb'))
        except:
            self.entity2hop_dic = {}
        self.word_2_frequency = self.LoadWord2Index('../../token2vec/SouGou_word_frequece/SogouLabDic.dic')
        self.not_pos = {'f','d','h','k','r','c','p','u','y','e','o','g','w','m'}  # 'q','mq','v','a','t',
        self.segger = thulac.thulac()
        self.pass_mention_dic = {'是什么','在哪里','哪里','什么','提出的','有什么','国家','哪个','所在',
                                 '培养出','为什么','什么时候','人','你知道','都包括','是谁','告诉我','又叫做','有','是'}
        self.fp = cs.open('../data/record/entity_extractor_ans.txt','w')
        print ('entity extractor loaded')
    
    def LoadWord2Index(self,path):
        dic = {}
        with cs.open(path,'r','utf-8') as fp:
            lines = fp.read().split('\n')[:-1]
            for line in lines:
                line = line.strip()
                token = line.split('\t')[0]
                f = int(line.split('\t')[1])//10000
                dic[token] = f
        return dic
    
    def get_mention_feature(self,question,mention):
        
        f1 = float(len(mention))#mention的长度
        
        try:
            f2 = float(self.word_2_frequency[mention])#mention的tf/10000
        except:
            f2 = 1.0
        if mention[-2:] == '大学':
            f2 = 1.0

        try:
            f3 = float(question.index(mention))
        except:
            f3 = 3.0
            #print ('这个mention无法提取位置')
        return [mention,f1,f2,f3]
    
    def extract_subject(self,entity_mentions,subject_props,question):
        '''
        根据前两部抽取出的实体mention和属性值mention，得到候选主语实体
        Input:
            entity_mentions: {str:list} {'贝鲁奇':['贝鲁奇',1,1,1]}
            subject_props: {str:list} {'1997-02-01':['1997年2月1日',1,1,1]}
        output:
            candidate_subject: {str:list}
        '''
        candidate_subject = {}
        
        for mention in entity_mentions:#遍历每一个mention
            #过滤词性明显不对的mention
            poses = self.segger.cut(mention)
            if len(poses) == 1 and poses[0][1] in self.not_pos:
                continue
            #过滤停用词
            if mention in self.pass_mention_dic:
                continue
            
            print ('====当前实体mention为：%s===='%(mention))
            if mention in self.mention2entity_dic:#如果它有对应的实体
                for entity in self.mention2entity_dic[mention]:
                    #mention的特征
                    mention_features = self.get_mention_feature(question,mention)
                    #得到实体两跳内的所有关系
                    entity = '<'+entity+'>'
                    if entity in self.entity2hop_dic:
                        relations = self.entity2hop_dic[entity]
                    else:            
                        relations = GetRelations_2hop(entity)
                        self.entity2hop_dic[entity] = relations
                    #计算问题和主语实体及其两跳内关系间的相似度
                    similar_features = ComputeEntityFeatures(question,entity,relations)
                    #实体的流行度特征
                    popular_feature = GetRelationNum(entity)
                    candidate_subject[entity] = mention_features + similar_features + [popular_feature ** 0.5]


        for prop in subject_props:
            print ('====当前属性mention为：%s===='%(prop))
            prop_mention = subject_props[prop]
            if prop_mention in self.pass_mention_dic or prop in self.pass_mention_dic:
                continue
            poses = self.segger.cut(prop_mention)
            if len(poses) == 1 and poses[0][1] in self.not_pos:
                continue

            #这里是否需要过滤？？？
            entity_prop = '<' + prop + '>'
            if entity_prop in candidate_subject:
                continue

            
            #mention的特征
            mention_features = self.get_mention_feature(question,prop_mention)
            #得到实体两跳内的所有关系
            entity = '\"'+prop+'\"'
            if entity in self.entity2hop_dic:
                relations = self.entity2hop_dic[entity]
            else:            
                relations = GetRelations_2hop(entity)
                self.entity2hop_dic[entity] = relations
            #计算问题和主语实体及其两跳内关系间的相似度
            similar_features = ComputeEntityFeatures(question,entity,relations)
            #实体的流行度特征
            popular_feature = GetRelationNum(entity)
            candidate_subject[entity] = mention_features + similar_features + [popular_feature ** 0.5]
        pickle.dump(self.entity2hop_dic,open('../data/entity2hop_dic.pkl','wb'))
        return candidate_subject
                    
    def GetCandidateEntity(self,corpus):
        true_num = 0.0
        one_num= 0.0
        one_true_num = 0.0
        subject_num = 0.0
        for i in range(len(corpus)):
            dic = corpus[i]
            question = dic['question']
            gold_entitys = dic['gold_entitys']
            # candidate_entity = {}
            print ('\n')
            print (i)
            print (question)
            starttime = time.time()
            #得到当前问题的候选主语mention和属性
            entity_mentions = dic['entity_mention']
            subject_props = dic['subject_props']
            candidate_entity = self.extract_subject(entity_mentions,subject_props,question)
            subject_num += len(candidate_entity)
            dic['candidate_entity'] = candidate_entity
            print('候选实体为：')
            for c in candidate_entity:
              print(c,candidate_entity[c])
            print (len(candidate_entity))
            print ('耗费时间%.2f秒'%(time.time()-starttime))
            
            if len(set(gold_entitys)) == len(set(gold_entitys).intersection(set(candidate_entity))):
                true_num +=1
                if len(gold_entitys) == 1:
                    one_true_num += 1
            else:
                print (question)
                print (gold_entitys)
                print (candidate_entity.keys)
                self.fp.write(str(i)+question+'\n')
                self.fp.write('\t'.join(gold_entitys)+'\n')
                self.fp.write('\t'.join(list(candidate_entity.keys()))+'\n\n')
            if len(gold_entitys) == 1:
                one_num += 1
                
        pickle.dump(self.entity2hop_dic,open('../data/entity2hop_dic.pkl','wb'))
        print ('单实体问题可召回比例为%.2f'%(one_true_num/one_num))
        print ('所有问题可召回比例为%.2f'%(true_num/len(corpus)))
        print ('平均每个问题的候选主语个数为:%.2f'%(subject_num/len(corpus)))
        return corpus

if __name__ == '__main__':
    inputpaths = ['../data/all_mentions_train.pkl','../data/all_mentions_valid.pkl','../data/all_mentions_test.pkl']
    outputpaths = ['../data/candidate_entitys_train.pkl','../data/candidate_entitys_valid.pkl','../data/candidate_entitys_test.pkl']
    se = SubjectExtractor()
    for i in range(len(inputpaths)):
        inputpath = inputpaths[i]
        outputpath = outputpaths[i]
        corpus = pickle.load(open(inputpath,'rb'))
        corpus = se.GetCandidateEntity(corpus)
        pickle.dump(corpus,open(outputpath,'wb'))
    se.fp.close()