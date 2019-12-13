 # -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:11:15 2019

@author: cmy
"""
import jieba
import codecs as cs
import pickle
import time
import numpy as np
from keras_bert import Tokenizer,get_custom_objects
from keras.models import load_model

class MentionExtractor(object):
    def __init__(self,):
        #分词词典加载
        with cs.open('../data/segment_dic.txt','r','utf-8') as fp:
            segment_dic = {}
            for line in fp:
                if line.strip():
                    segment_dic[line.strip()] = 0
        self.segment_dic = segment_dic
        self.max_seq_len = 20
        begin = time.time()
        jieba.load_userdict('../data/segment_dic.txt')
        print ('加载用户分词词典时间为:%.2f'%(time.time()-begin))
        #加载训练好的实体识别模型
        custom_objects = get_custom_objects()
        self.ner_model = load_model('../data/model/ner_model.h5', custom_objects=custom_objects)
        #加载bert tokenlizer
        dict_path = '../../news_classifer_task/wwm/vocab.txt'
        token_dict = {}
        with cs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        self.tokenizer = Tokenizer(token_dict)
        print ('mention extractor loaded')

    
    def extract_mentions(self,question):
        '''
        返回字典，实体mentions
        '''
        entity_mention = {}
        
        #使用jieba粗糙分词的方式得到候选mention
        mentions = []
        tokens = jieba.lcut(question)
        for t in tokens:
            if t in self.segment_dic:
                mentions.append(t)
        
        #使用序列标注模型来抽取候选 mention
        x1, x2 = self.tokenizer.encode(first=question,max_len = self.max_seq_len)
        x1,x2= np.array([x1]),np.array([x2])
        predict_y = self.ner_model.predict([x1,x2],batch_size=32).tolist()[0]#(len,1)
        predict_y = [1 if each[0]>0.5 else 0 for each in predict_y]
        mentions_bert = self.restore_entity_from_labels(predict_y,question)
        
        #判断是否属于mention_dic
        mentions = mentions + mentions_bert
        for token in mentions:
            entity_mention[token] = token

        return entity_mention
    
    def restore_entity_from_labels(self,labels,question):
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
        
    def GetEntityMention(self,corpus):
        mention_num = 0
        for i in range(len(corpus)):
            dic = corpus[i]
            question = dic['question']
            dic['entity_mention'] = self.extract_mentions(question)
            corpus[i] = dic
            print (question)
            print (dic['entity_mention'])
        print(mention_num,len(corpus))
        return corpus


        
if __name__ == '__main__':
    inputpaths = ['../data/corpus_train.pkl','../data/corpus_valid.pkl','../data/corpus_test.pkl']
    outputpaths = ['../data/entity_mentions_train.pkl','../data/entity_mentions_valid.pkl','../data/entity_mentions_test.pkl']
    corpuses = []
    me = MentionExtractor()
    for i in range(len(inputpaths)):
        inputpath = inputpaths[i]
        outputpath = outputpaths[i]
        corpus = pickle.load(open(inputpath,'rb'))
        corpus = me.GetEntityMention(corpus)
        print ('得到实体mention')
        pickle.dump(corpus,open(outputpath,'wb'))
    