 # -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:11:15 2019

@author: cmy
"""
import jieba
import codecs as cs
import pickle
import time
import chardet
class MentionExtractor(object):
    def __init__(self,):
        with cs.open('../data/segment_dic.txt','r','utf-8') as fp:
            segment_dic = {}
            for line in fp:
                if line.strip():
                    segment_dic[line.strip()] = 0
        self.segment_dic = segment_dic
        
        begin = time.time()
        jieba.load_userdict('../data/segment_dic.txt')
        self.question2mention = pickle.load(open('../data/question_2_mention.pkl','rb'))
        print ('加载用户分词词典时间为:%.2f'%(time.time()-begin))
        print ('mention extractor loaded')

    
    def extract_mentions(self,question):
        '''
        返回字典，实体mentions
        '''
        entity_mention = {}
        
        #使用jieba分词的方式得到候选mention
        mentions = []
        tokens = jieba.lcut(question)
        for t in tokens:
            if t in self.segment_dic:
                mentions.append(t)
        #使用基于bert的序列标注模型得到候选mention,这里是直接把整个语料集中问题对应的mention都保存了下来，每次直接调文件，所以对新问题不适用，需要进一步改善
#        try:
#            for m in self.question2mention[question][0]:
#                if m not in mentions:
#                    mentions.append(m)
#            for m in self.question2mention[question][1]:
#                if m not in mentions:
#                    mentions.append(m)
#        except:
#            print ('this question dont have bert mention')
        for token in mentions:
            entity_mention[token] = token

        return entity_mention
    
        
        
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
    inputpaths = ['../data/corpus_train.pkl','../data/corpus_valid.pkl']
    outputpaths = ['../data/entity_mentions_train.pkl','../data/entity_mentions_valid.pkl']
    corpuses = []
    me = MentionExtractor()
    for i in range(len(inputpaths)):
        inputpath = inputpaths[i]
        outputpath = outputpaths[i]
        corpus = pickle.load(open(inputpath,'rb'))
        corpus = me.GetEntityMention(corpus)
        print ('得到实体mention')
        pickle.dump(corpus,open(outputpath,'wb'))
    