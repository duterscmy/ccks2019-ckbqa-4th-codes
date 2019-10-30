# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:51:55 2019

@author: cmy
"""
import codecs as cs

mention2entity_dic = {}
with cs.open('../PKUBASE/pkubase-mention2ent.txt','r','utf-8') as fp:
    mention2entity_dic = {}
    lines = fp.read().split('\n')[0:-1]
    for line in lines:
        if line.strip():
            mention = line.split('\t')[0]
            entity = line.split('\t')[1]
            if mention in mention2entity_dic:
                mention2entity_dic[mention].append(entity)
            else:
                mention2entity_dic[mention]  = [entity]
                
import pickle
pickle.dump(mention2entity_dic,open('../data/mention2entity_dic.pkl','wb'))
                