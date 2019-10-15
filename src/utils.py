# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:52:18 2019

@author: cmy
"""

import thulac
import codecs as cs
import numpy as np
import tensorflow as tf
segger = thulac.thulac(seg_only=True)

#word embedding
chinese_embedding = {}
with cs.open('../../token2vec/zhwiki_2017_03.sg_50d.word2vec','r','utf-8') as fp:
    lines = fp.read().split('\n')[1:300000]
    for line in lines:
        line = line.strip()
        elements = line.split(' ')
        chinese_embedding[elements[0]] = []
        for num in elements[1:]:
            chinese_embedding[elements[0]].append(float(num))
print ('wordvec loaded')


def ComputeSimilar(p_tokens,q_tokens,wordvec):
    '''
    输入两个词序列，计算最大余弦距离和最大点乘值
    input:
        p_tokens: python-list
        q_tokens: python-list
    '''
    def cosine_Matrix(A, B):
        AB = np.matmul(A,np.transpose(B))
        A_norm = np.sqrt(np.sum(np.multiply(A,A),axis=-1))
        B_norm = np.sqrt(np.sum(np.multiply(B,B),axis=-1))
        norm = np.matmul(np.expand_dims(A_norm,axis=1),np.transpose(np.expand_dims(B_norm,axis=1)))
        return np.divide(AB,norm)
    p_embeddings = []
    q_embeddings = []
    for p in p_tokens:
        try:
            p_embeddings.append(wordvec[p])
        except:
            pass
    for q in q_tokens:
        try:
            q_embeddings.append(wordvec[q])
        except:
            pass
    if len(p_embeddings) == 0 or len(q_embeddings) == 0:
        return 0.0
    #计算余弦距离
    matrix = cosine_Matrix(np.array(p_embeddings),np.array(q_embeddings))
    sim_cos= np.sum(np.max(matrix,axis=1))#cos相似度之和
    return sim_cos


def ComputeTupleFeatures(predicates,question):
    '''
    为每个候选tuple和问题计算人工特征
    predicates:[r1name,r2name]或[r1name,r2name]
    question:str
    q_tokens:未加载词典的分词结果
    q_chars:分字结果
    '''
    p_tokens = []
    for p in predicates:
        p_tokens.extend(segger.cut(p))
    p_tokens = [token[0] for token in p_tokens]
    p_chars = [char for char in ''.join(predicates)]
    
    q_tokens = segger.cut(question)
    q_tokens = [token[0] for token in q_tokens]
    q_chars = [char for char in question]
    #计算谓词和问题的word overlap
    word_overlap = len(set(p_tokens).intersection(set(q_tokens)))
    #计算谓词和问题的char overlap
    char_overlap = len(set(p_chars).intersection(set(q_chars)))
    #向量序列相似度
    word_similar_cos= ComputeSimilar(p_tokens,q_tokens,chinese_embedding)
    char_similar_cos= ComputeSimilar(p_chars,q_chars,chinese_embedding)
    return [word_overlap,word_similar_cos,char_overlap,char_similar_cos]

def ComputeEntityFeatures(question,entity,relations):
    '''
    抽取每个实体或属性值2hop内的所有关系，来跟问题计算各种相似度特征
    input:
        question: python-str
        entity: python-str <entityname>
        relations: python-dic key:<rname>
    output：
        [word_overlap,char_overlap,word_embedding_similarity,char_overlap_ratio]
    '''
    #得到主语-谓词的tokens及chars
    if entity[0] == '<' or entity[0] == '\"':
        entity = entity[1:-1]
    p_tokens = []
    for p in relations:
        p_tokens.extend(segger.cut(p[1:-1]))
    p_tokens.extend(segger.cut(entity))
    p_tokens = [token[0] for token in p_tokens]
    p_chars = [char for char in ''.join(p_tokens)]
    
    q_tokens = segger.cut(question)
    q_tokens = [token[0] for token in q_tokens]
    q_chars = [char for char in question]
    #计算谓词和问题的word overlap
    word_overlap = len(set(p_tokens).intersection(set(q_tokens)))
    #计算谓词和问题的char overlap
    char_overlap = len(set(p_chars).intersection(set(q_chars)))
    #实体名和问题的overlap除以实体名长度的比例
    return [word_overlap,char_overlap]

if __name__ == '__main__':
    print (ComputeTupleFeatures(['中文名'],'高谭市的守护者的中文名是什么？'))
    print (ComputeTupleFeatures(['狗'],'高谭市的守护者的中文名是什么？'))
    print (ComputeTupleFeatures(['曹'],'高谭市的守护者的中文名是什么？'))
