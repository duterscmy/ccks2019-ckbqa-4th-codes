# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:40:56 2019

@author: cmy
"""
segment_dic = {}

with open('../PKUBASE/pkubase-full.txt','r',encoding='utf-8') as fp:
    for line in fp:
        if line:
            entity = line.split('\t')[0]
            ename = entity[1:-1]
            if '_' in ename:
                ename = ename.split('_')[0]
            segment_dic[ename] = 1
with open('../PKUBASE/pkubase-mention2ent.txt','r',encoding='utf-8') as fp:
    for line in fp:
        if line:
            ename = line.split('\t')[0]
            segment_dic[ename] = 1
            
with open('../data/segment_dic.txt','w',encoding='utf-8') as fp:
    for s in segment_dic:
        if s:
            fp.write(s+'\n')

with open('../data/segment_dic.txt','r',encoding = 'utf-8') as fp:
    content = fp.read()

with open('../data/segment_dic.txt','w',encoding = 'utf-8') as fp:
    fp.write(content[:-1])
        
prop_dic = {}
with open('../PKUBASE/pkubase-full.txt','r',encoding='utf-8') as fp:
    for line in fp:
        if line:
            try:
                ob = line.split('\t')[2][:-3]
                if ob[0] == '\"':
                    if ob[1:-1] in prop_dic:
                        #print (ob[1:-1])
                        prop_dic[ob[1:-1]] += 1
                    else:
                        prop_dic[ob[1:-1]] = 1
            except:
                continue
            

import pickle 
pickle.dump(prop_dic,open('../data/prop_dic.pkl','wb'))