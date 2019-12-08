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
        
        self.me = MentionExtractor()
        self.pe = PropExtractor()
        self.se = SubjectExtractor()
        self.te = TupleExtractor()
        self.topn_e = 6
        self.topn_t = 3
        
        self.subject_classifer_model = pickle.load(open('../data/model/entity_classifer_model.pkl','rb'))
        self.tuple_classifer_model = pickle.load(open('../data/model/tuple_classifer_model.pkl','rb'))
        self.tuple_scaler = joblib.load('../data/tuple_scaler')
        
        self.segger = thulac.thulac()
        self.not_relation = {'<中文名>','<外文名>','<本名>','<别名>','<国籍>','<职业>'}#双实体问题桥接不考虑的关系
        self.validAns = {}
        self.qindex = 1
        
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
            features.append(tuples[t][2:])
        xxx = self.tuple_scaler.transform(np.array(features))
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
    
    def tuple_filter_by_nn(self,question,candidate_tuples):
        '''
        输入候选答案，将其填补到20个，使用训练好的文本匹配模型进行打分，返回分数最高的候选答案
        '''
        candidate_list = [t for t in candidate_tuples]
        if len(candidate_list)<(self.topn_t):
            repeat_list = [t for t in candidate_list]
            repeat_num = self.topn_t//len(candidate_list)+1
            for j in range(repeat_num):
                candidate_list = candidate_list + repeat_list
        tuples = candidate_list[:self.topn_t]
        
        q_inputs,q_len,e_inputs,e_len,r_inputs,r_len,f = self.rep.Generate_batch(question,tuples,candidate_tuples)#(num_candi,num_feature)
        
        predict_index = self.sess.run(self.Model.prediction,
                     feed_dict = { self.Model.q_input:q_inputs, 
                                  self.Model.e_input:e_inputs, 
                                  self.Model.r_input:r_inputs,
                                  self.Model.q_len:q_len,
                                  self.Model.e_len:e_len,
                                  self.Model.r_len:r_len,
                                  self.Model.features:f})
        return tuples[predict_index]
    
    def add_props(self,entity_mention,pred_props):
        '''
        用entity mention对props做一个补充
        '''
        #补充属性值里带顿号的情况
                           
        subject_props = {}
        subject_props.update(pred_props['mark_props'])
        subject_props.update(pred_props['time_props'])
        subject_props.update(pred_props['digit_props'])
        subject_props.update(pred_props['other_props'])
        subject_props.update(pred_props['fuzzy_props'])
        
        special_props = {}
        subject_props.update(pred_props['mark_props'])
        subject_props.update(pred_props['time_props'])
        
        return subject_props,special_props

    def correct(self,question,tuples):
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
    
    def Answer_By_KB(self,question):
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
        if self.qindex<=446:
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
            print ('×该问题无法生成候选答案')
            return []
        
        tuples = self.tuple_filter(tuples)#得到top3的单实体问题tuple
        dic['tuples_filter'] = tuples
        print ('====筛选后的候选查询路径为====')
        print (tuples)
        
        #从相似度前三的候选tuple中选择和问题重叠字数最多的
        top_tuple = self.correct(question,tuples)
        #尝试桥接双实体路径
        twoEntityTuple = self.GetTwoEntityTuple(question,subjects,dic['tuples'])
        if len(twoEntityTuple)>0:
            top_tuple = twoEntityTuple
                
                
        dic['top_tuple'] = top_tuple
        print ('====最终候选查询路径为====')
        print (top_tuple)
        
        #将tuples中的属性值变为无引号形式
        search_paths = [ele for ele in top_tuple]
        #生成cypher语句并查询
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
        self.validAns[self.qindex] =dic
        self.qindex += 1
        return answer

        
    def ProcessTestCorpus(self,path):
        with cs.open(path,'r','utf-8') as fp:
            lines = fp.read().split('\r\n')[:-1]
        answers = []
        for line in lines:
            question = ''.join(line.split(':')[1:])
            answer = self.Answer_By_KB(question)
            answers.append(answer)
        return answers

if __name__ == '__main__':
    ANS = AnswerByPkubase()
    test_answers = ANS.ProcessTestCorpus('../corpus/task6ckbqa_test.questions.txt')
    text = []
    with cs.open('../data/record/test_answer_last.txt','w','utf-8') as fp:#这个文件是神经网络方法最终得到的可提交结果
        for ans in test_answers:
            text.append('\t'.join(ans))
        fp.write('\n'.join(text))
    pickle.dump(ANS.te.sentencepair2sim,open('../data/sentencepair2sim_dic.pkl','wb'))
    pickle.dump(ANS.validAns,open('../data/TestAns_last.pkl','wb'))