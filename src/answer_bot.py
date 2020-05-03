# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:17:41 2019

@author: Administrator
"""

import codecs as cs
import pickle
import numpy as np
from mention_extractor import MentionExtractor
from prop_extractor import PropExtractor
from entity_extractor import SubjectExtractor
from tuple_extractor import TupleExtractor
from kb import session,GetTwoEntityTuple
from sklearn.externals import joblib
import re
import thulac

class AnswerByPkubase(object):
    def __init__(self,):
        
        
        self.te = TupleExtractor()
        self.me = MentionExtractor()
        self.se = SubjectExtractor()
        self.pe = PropExtractor()
        self.topn_e = 5
        self.topn_t = 3
        
        self.subject_classifer_model = pickle.load(open('../data/model/entity_classifer_model.pkl','rb'))
        self.tuple_classifer_model = pickle.load(open('../data/model/tuple_classifer_model.pkl','rb'))
        
        self.segger = thulac.thulac()
        self.not_relation = {'<中文名>','<外文名>','<本名>','<别名>','<国籍>','<职业>'}#双实体问题桥接不考虑的关系
        
    def LoadMentionDic(self):
        with cs.open('../PKUBASE/pkubase-mention2ent.txt','r','utf-8') as fp:
            mention2entity_dic = {}
            lines = fp.read().split('\n')[1:-5]
            for line in lines:
                if line.strip():
                    mention = line.split('\t')[0]
                    entity = line.split('\t')[1]
                    if mention in mention2entity_dic:
                        mention2entity_dic[mention].append(entity)
                    else:
                        mention2entity_dic[mention]  = [entity]
        return mention2entity_dic
        
    def subject_filter(self,subjects):
        '''
        输入候选主语和对应的特征，使用训练好的模型进行打分，排序后返回前topn个候选主语
        '''
        entitys = []
        features = []
        for s in subjects:
            entitys.append(s)
            features.append(subjects[s][1:])
        prepro = self.subject_classifer_model.predict_proba(np.array(features))[:,1].tolist()
        sample_prop = [each for each in zip(prepro,entitys)]#(prop,(tuple))
        sample_prop = sorted(sample_prop, key=lambda x:x[0], reverse=True)
        entitys = [each[1] for each in sample_prop]
        if len(entitys)>self.topn_e:
            predict_entitys = entitys[:self.topn_e]
        else:
            predict_entitys = entitys
        new_subjects = {}
        for e in predict_entitys:
            new_subjects[e] = subjects[e]
        return new_subjects
    
    def tuple_filter(self,tuples):
        '''
        输入候选答案和对应的特征，使用训练好的模型进行打分，排序后返回前topn个候选答案
        '''
        tuple_list = []
        features = []
        for t in tuples:
            tuple_list.append(t)
            features.append(tuples[t][-1:])
        #xxx = self.tuple_scaler.transform(np.array(features))
        xxx = features
        prepro = self.tuple_classifer_model.predict_proba(xxx)[:,1].tolist()
        sample_prop = [each for each in zip(prepro,tuple_list)]#(prop,(tuple))
        sample_prop = sorted(sample_prop, key=lambda x:x[0], reverse=True)
        tuples_sorted = [each[1] for each in sample_prop]
        if len(tuples_sorted) > self.topn_t:
            tuples_sorted = tuples_sorted[:self.topn_t]
        return tuples_sorted
    
    def GetTwoEntityTuple(self,question,subjects,tuples):
        #得到相似度排名前十的简单tuple
        tuple_with_prop = []
        for t in tuples:
            tuple_with_prop.append([t,tuples[t][-1]])
        tuple_with_prop = sorted(tuple_with_prop, key=lambda x:x[1], reverse=True)#排序
        tuple_with_prop_len2 = []
        for t in tuple_with_prop:
            if len(t[0]) ==2 and t[0][1] not in self.not_relation:
                tuple_with_prop_len2.append(t)
        #sorted_tuples = [each[0] for each in tuple_with_prop][:3]
        sorted_tuples_len2 = [each[0] for each in tuple_with_prop_len2]
        sorted_tuples_len2 = sorted_tuples_len2[:20] if len(sorted_tuples_len2)>20 else sorted_tuples_len2
        
        #从单关系tuple中，选择出能和其他主语进行桥接的，并生成新的候选答案
        new_tuples = []
        for simple_tuple in sorted_tuples_len2:
            entity = simple_tuple[0]
            for other_subject in subjects:
                e1_mention = re.sub('_|（|）','',entity[1:-1])
                e2_mention = re.sub('_|（|）','',other_subject[1:-1])
                if entity != other_subject and len(set(e1_mention).intersection(set(e2_mention)))==0:
                    two_entity_tuple = GetTwoEntityTuple(simple_tuple[0],simple_tuple[1],other_subject)
                    two_entity_tuple_without_special_relation = []
                    for t in two_entity_tuple:
                        if t[1] not in self.not_relation and t[2] not in self.not_relation:
                            two_entity_tuple_without_special_relation.append(t)
                    new_tuples.extend(two_entity_tuple_without_special_relation)
        #(e1,r1,r2,e2)
        #从新的tuples里选择r2 overlap最高的
        max_ = 0
        if len(new_tuples) == 0:
            two_entity_tuple = ()
        for t in new_tuples:
            overlap = len(set(t[0][1:-1]+t[1][1:-1]+t[2][1:-1]+t[3][1:-1]).intersection(set(question)))
            if overlap>max_:
                max_ = overlap
                two_entity_tuple = t
        return two_entity_tuple
    
    
    def add_props(self,entity_mention,pred_props):
        #所有可能的属性值主语                         
        subject_props = {}
        subject_props.update(pred_props['mark_props'])
        subject_props.update(pred_props['time_props'])
        subject_props.update(pred_props['digit_props'])
        subject_props.update(pred_props['other_props'])
        subject_props.update(pred_props['fuzzy_props'])
        
        #时间及称号类的属性值
        special_props = {}
        special_props.update(pred_props['mark_props'])
        special_props.update(pred_props['time_props'])
        
        return subject_props,special_props

    def get_most_overlap_tuple(self,question,tuples):
        #从排名前几的tuples里选择与问题overlap最多的
        max_ = 0
        ans = tuples[0]
        for t in tuples:
            text = ''
            for element in t:
                element = element[1:-1].split('_')[0]
                text = text + element
            f1 = len(set(text).intersection(set(question)))
            f2 = len(set(text).intersection(set(question)))/len(set(text))
            f = f1+f2
            if f>max_:
                ans = t
                max_ = f
        return ans
    
    def answer_main(self,question):
        '''
        输入问题，依次执行：
        抽取实体mention、抽取属性值、生成候选实体并得到特征、候选实体过滤、生成候选查询路径（单实体双跳）、候选查询路径过滤
        使用top1的候选查询路径检索答案并返回
        input:
            question : python-str
        output:
            answer : python-list, [str]
        '''
        dic = {}
        question = re.sub('在金庸的小说《天龙八部》中，','',question)
        question = re.sub('电视剧武林外传里','',question)
        question = re.sub('《射雕英雄传》里','',question)
        question = re.sub('情深深雨濛濛中','',question)
        question = re.sub('《.+》中','',question)
        question = re.sub('常青藤大学联盟中','',question)
        question = re.sub('原名','中文名',question)
        question = re.sub('英文','外文',question)
        question = re.sub('英语','外文',question)
        dic['question'] = question
        print (question)
        
        mentions = self.me.extract_mentions(question)
        dic['mentions'] = mentions
        print ('====实体mention为====')
        print (mentions.keys())
        
        props= self.pe.extract_properties(question)
        subject_props,special_props = self.add_props(mentions,props)
        dic['props'] = subject_props
        print ('====属性mention为====')
        print (subject_props.keys())
        
        subjects = self.se.extract_subject(mentions,subject_props,question)
        dic['subjects'] = subjects
        print ('====主语实体为====')
        print (subjects.keys())
        if len(subjects) == 0:
            return []
        
        subjects= self.subject_filter(subjects)
        #过滤后仍然加上特殊的时间/称号/书名类属性值
        for prop in special_props:
            sub = '\"'+prop+'\"'
            if sub not in subjects:
                subjects[sub] = [special_props[prop],3,1,1,2,6]
        dic['subjects_filter'] = subjects
        print ('====筛选后的主语实体为====')
        print (subjects.keys())
        if len(subjects) == 0:
            return []
        
        tuples = self.te.extract_tuples(subjects,question)
        dic['tuples'] = tuples
        if len(tuples) == 0:
            return []
        
        tuples = self.tuple_filter(tuples)#得到top1的单实体问题tuple
        dic['tuples_filter'] = tuples
        print ('====筛选后的候选查询路径为====')
        print (tuples)
        
        #top_tuple = tuples[0]
        top_tuple = self.get_most_overlap_tuple(question,tuples)   
#        twoEntityTuple = self.GetTwoEntityTuple(question,subjects,dic['tuples'])
#        if len(twoEntityTuple)>0:
#            top_tuple = twoEntityTuple
                

        print ('====最终候选查询路径为====')
        print (top_tuple)
        
        
        #生成cypher语句并查询
        search_paths = [ele for ele in top_tuple]
        if len(search_paths) == 2:
            sql = "match (a:Entity)-[r1:Relation]-(b) where a.name=$ename and r1.name=$rname return b.name"
            res = session.run(sql,ename=search_paths[0],rname=search_paths[1])
            ans = [record['b.name'] for record in res]
        elif len(search_paths) == 3:
            sql = "match (a:Entity)-[r1:Relation]-()-[r2:Relation]-(b) where a.name=$ename and r1.name=$rname1 and r2.name=$rname2 return b.name"
            res = session.run(sql,ename=search_paths[0],rname1=search_paths[1],rname2=search_paths[2])
            ans = [record['b.name'] for record in res]
        elif len(search_paths) == 4:
            sql = "match (a:Entity)-[r1:Relation]-(c)-[r2:Relation]-(b:Entity) where a.name=$ename1 and r1.name=$rname1 and r2.name=$rname2 and b.name=$ename2 return c.name"
            res = session.run(sql,ename1=search_paths[0],rname1=search_paths[1],rname2=search_paths[2],ename2=search_paths[3])
            ans = [record['c.name'] for record in res]
        else:
            print ('这个查询路径不规范')
            ans = []
        #将答案中的属性值还原
        answer = ans
        dic['answer'] = answer
        print ('====答案为====')
        print (answer)
        print ('\n')
        return answer

        
    def add_answers_to_corpus(self,corpus):
        
        for i in range(len(corpus)):
            sample = corpus[i]
            question = sample['question']
            ans = self.answer_main(question)
            sample['predict_ans'] = ans
        return corpus

if __name__ == '__main__':
    ansbot = AnswerByPkubase()
    corpus = pickle.load(open('../data/candidate_entitys_filter_test.pkl','rb'))
    corpus = ansbot.add_answers_to_corpus(corpus)
    #pickle.dump(ansbot.te.sentencepair2sim,open('../data/sentencepair2sim_dic.pkl','wb'))
    
    ave_f = 0.0
    for i in range(len(corpus)):
        sample = corpus[i]
        gold_ans = sample['answer']
        pre_ans = sample['predict_ans']
        true = len(set(gold_ans).intersection(set(pre_ans)))
        p = true / len(set(pre_ans))
        r = true / len(set(gold_ans))
        try:
            f = 2*p*r/(p+r)
        except:
            f = 0.0
        ave_f += f
    ave_f /= len(corpus)
    
    print (ave_f)