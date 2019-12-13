# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:07:52 2019

@author: cmy
"""

import pickle 
import codecs as cs
import re
import thulac
import time


class PropExtractor(object):
    def __init__(self):
        self.prop_dic = pickle.load(open('../data/prop_dic.pkl','rb'))#键没有引号
        self.char_2_prop = pickle.load(open('../data/char_2_prop.pkl','rb'))
        self.segger = thulac.thulac()
        print ('prop extractor loaded')
        
    
    
    def extract_properties(self,question):
        '''
        输入一个问题，抽取出所有能和知识库中的属性值匹配的字符串，筛选后返回
        input:
            question : python-str
        output:
            props : python-dic
        '''
        props = {}#键为知识库里prop，值为mention
        QUES = question
        
        #包含在双引号 书名号里的属性
        mark_props = {}
        elements = re.findall('\".+\"|《.+》',question)
        if len(elements)>0:
            for e in elements:  # '甲天下', '完美的搜索引擎，'
                if e in self.prop_dic:  # 一般书名号的属性就是需要的属性
                    mark_props[e] = e
                question = re.sub(e, '', question)
        props['mark_props'] = mark_props
        
        #时间属性
        time_props = {}
        #提取年月日
        year_month_day = re.findall('\d+年\d+月\d+日|\d+年\d+月\d+号|\d+\.\d+\.\d+',question)
        for ymd in year_month_day:
            rml_norm = self.TransNormalTime(ymd)
            time_props[rml_norm] = ymd
            question = re.sub(ymd,'',question)
        #提取月日
        month_day = re.findall('\d+月\d+日|\d+月\d+号|\d+年\d+月',question)
        for ymd in month_day:
            rml_norm = self.TransNormalTime(ymd)
            time_props[rml_norm] = ymd
            question = re.sub(ymd,'',question)
        #提取年份
        years = re.findall('\d+年',question)
        for ymd in years:
            rml_norm = self.TransNormalTime(ymd)
            time_props[rml_norm] = ymd
            question = re.sub(ymd,'',question) 
        props['time_props'] = time_props
        #数字属性
        digit_props = {}
        elements = re.findall('\d+',question)
        if len(elements)>0:
            for e in elements:
                if e in self.prop_dic:
                    digit_props[e] = e
        props['digit_props'] = digit_props

        
        #其他属性,去重
        other_props = {}
        length = len(question)
        props_ngram = []
        max_len = 0
        for l in range(length,0,-1):#只考虑长度大于1的可匹配属性值
            for i in range(length-l+1): 
                if question[i:i+l] in self.prop_dic:
                    props_ngram.append(question[i:i+l])
                    if len(question[i:i+l])>max_len:
                        max_len = len(question[i:i+l])
                        
        stop_props = []
        for p in props_ngram:
            for q in props_ngram:
                if p in q and p!=q and self.segger.cut(p)[0][1] not in ['ns']:  # 加拿大的，台湾的等问题 p不是地名
                    stop_props.append(p)
                    
        new_props = []  # 去掉包含在更长属性值中的属性值
        for p in props_ngram:
            if p not in stop_props:
                new_props.append(p)
                    
        new_new_props = []  # 去掉长度过于短的属性值
        for p in new_props:
            if len(p) == 1 and self.segger.cut(p)[0][1] in ['n']:  # 单字名词
                new_new_props.append(p)
            elif (len(p) >= (max_len * 0.5) and len(p) != 1) or self.segger.cut(p)[0][1] in ['n', 'ns'] or self.exist_digit(p):  # 长度过短且词性名词比较重要
                new_new_props.append(p)
                
        for p in new_new_props:
            other_props[p] = p
        props['other_props'] = other_props
        
        #模糊匹配得到的属性
        stop_dic = {'有', '的', '是', '在', '上', '哪', '里', '\"', '什', '么', '中', '个'}
        prop2num = {}
        for char in QUES:
            if char in stop_dic:
                continue
            else:
                try:
                    for p in self.char_2_prop[char]:
                        if p in prop2num:
                            prop2num[p] += 1
                        else:
                            prop2num[p] = 1
                except:
                    continue
        sort_props = sorted(prop2num.items(),key = lambda prop2num:prop2num[1],reverse=True)
        top3_props = [key for key,value in sort_props[:3]]  # top3
        fuzzy_props = {}
        for p in top3_props:
            fuzzy_props[p] = p
        props['fuzzy_props'] = fuzzy_props  # 取与问题中匹配字数最多的属性作为候选

        return props
    
    def extract_subject_properties(self,question):
        '''
        输入一个问题，抽取出所有能和知识库中的属性值匹配的字符串，并将更有可能作为简单问题主语的属性值提取出来
        input:
            question : python-str
        output:
            props : python-dic
        '''
        pred_props = self.extract_properties(question)
        if len(pred_props['mark_props'])!=0:
            subject_props = pred_props['mark_props']
        elif len(pred_props['time_props'])!=0:
            subject_props = pred_props['time_props']
        elif len(pred_props['digit_props'])!=0:
            subject_props = pred_props['digit_props']
        else:
            subject_props = pred_props['other_props']
            subject_props.update(pred_props['fuzzy_props'])
        return subject_props
    
    
    def GetProps(self,corpus):
        gold_num = 0
        true_num = 0
        entity_error = []
        irregular = []
        all_props_num = 0.0
        for i in range(len(corpus)):
            question = corpus[i]['question']
            gold_entitys = corpus[i]['gold_entitys']


            # 提取gold props
            gold_props = []
            for x in gold_entitys:
                if x[0] == '\"':
                    gold_props.append(x)
            
            # 得到抽取出的属性字典并保存
            pred_props = self.extract_properties(question)  # 得到的均不包含引号
            corpus[i]['all_props'] = pred_props

            # 得到所有可能的属性corpus[i]['subject_props']
            subject_props = {}
            subject_props.update(pred_props['mark_props'])
            subject_props.update(pred_props['time_props'])
            subject_props.update(pred_props['digit_props'])
            subject_props.update(pred_props['other_props'])
            subject_props.update(pred_props['fuzzy_props'])
            corpus[i]['subject_props'] = subject_props
            all_props_num += len(corpus[i]['subject_props'])
            
            # 统计该模块抽取唯一主语实体的召回率
            if len(gold_props) == 1 and len(gold_entitys)==1:
                gold_num += 1
                if_same = self.CheckSame(gold_props,subject_props)  # 判断抽取出的属性值是否完全包括了gold props
                true_num += if_same
                if not if_same:
                    print ('主语属性值抽取失败')
                    entity_error.append(i)
                else:
                    print ('主语属性值抽取成功')
                print (i, question)
                print (gold_props)
                print (subject_props)
                print ('\n')
        print ('单主语且主语为属性值问题中，能找到所有主语属性值的比例为:%.2f'%(true_num/gold_num))
        print ('平均每个问题属性为:%.2f' % (all_props_num / len(corpus)))
        print (entity_error)
        print (irregular)
        return corpus
    
    def CheckSame(self,gold_props,pred_props):
        pred_props_list = []
        for p in pred_props:  # 取得是key键
            pred_props_list.append('\"'+p+'\"')
        join_props = set(pred_props_list).intersection(set(gold_props))
        if len(join_props) == len(gold_props):
            return 1
        else:
            return 0
        
    def exist_digit(self,p):
        '''
        判断字符串中是否存在数字
        '''
        for i in range(10):
            if str(i) in p:
                return 1
        return 0
    
    def TransNormalTime(self,time):
        digits = re.findall('\d+',time)
        elements = []
        for d in digits:
            if len(d)>2:
                elements.append(d)
            elif len(d) == 2:
                if int(d[0])>3:
                    elements.append('19'+ d)
                else:
                    elements.append(d)
            else:
                elements.append('0'+d)
        return '-'.join(elements)

if __name__ == "__main__":
    inputpaths = ['../data/entity_mentions_train.pkl','../data/entity_mentions_valid.pkl','../data/entity_mentions_test.pkl']
    outputpaths = ['../data/all_mentions_train.pkl','../data/all_mentions_valid.pkl','../data/all_mentions_test.pkl']
    starttime = time.time()
    pe = PropExtractor()
    for i in range(1,2):
        inputpath = inputpaths[i]
        outputpath = outputpaths[i]
        corpus = pickle.load(open(inputpath,'rb'))
        corpus = pe.GetProps(corpus)
        print ('得到实体mention')
        pickle.dump(corpus,open(outputpath,'wb'))
    print('耗费时间%.2f秒'%(time.time()-starttime))
