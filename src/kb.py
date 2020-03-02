# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:28:42 2019

@author: cmy
"""
from neo4j import GraphDatabase
import time

#neo4j
driver = GraphDatabase.driver("bolt://localhost:7687")
session = driver.session()
begintime = time.time()
session.run('MATCH (n) OPTIONAL MATCH (n)-[r]->() RETURN count(n.name) + count(r)')
#session.run('CREATE INDEX ON:Entity(name)')
endtime = time.time()
print ('start neo4j and match all entities,the time is %.2f'%(endtime-begintime))

def GetRelationPaths(entity):
    '''根据实体名，得到所有1跳2跳关系list,2d-list'''

    cql_1 = "match (a:Entity)-[r1:Relation]-() where a.name=$name return DISTINCT r1.name"
    cql_2 = "match (a:Entity)-[r1:Relation]-()-[r2:Relation]->() where a.name=$name return DISTINCT r1.name,r2.name"
    rpaths1 = []
    res = session.run(cql_1,name=entity)#一个多个record组成的集合
    for record in res:#每个record是一个key value的有序序列
        rpaths1.append([record['r1.name']])
    rpaths2 = []
    res = session.run(cql_2,name=entity)
    for record in res:
        rpaths2.append([record['r1.name'],record['r2.name']])
    return rpaths1+rpaths2

def GetRelationPathsSingle(entity):
    '''根据实体名，得到所有1跳2跳关系list,2d-list'''

    cql_1 = "match (a:Entity)-[r1:Relation]-() where a.name=$name return DISTINCT r1.name"
    rpaths1 = []
    res = session.run(cql_1,name=entity)#一个多个record组成的集合
    for record in res:#每个record是一个key value的有序序列
        rpaths1.append([record['r1.name']])
    return rpaths1

def GetRelations_2hop(entity):
    '''根据实体名，得到两跳内的所有关系dic'''
    cql= "match (a:Entity)-[r1:Relation]-()-[r2:Relation]->() where a.name=$name return DISTINCT r1.name,r2.name"
    rpaths2 = []
    res = session.run(cql,name=entity)
    for record in res:
        rpaths2.append([record['r1.name'],record['r2.name']])
    dic = {}
    for rpath in rpaths2:
        for r in rpath:
            dic[r] = 0
    return dic

def GetRelationNum(entity):
    '''根据实体名，得到与之相连的关系数量，代表实体在知识库中的流行度'''
    cql= "match p=(a:Entity)-[r1:Relation]-() where a.name=$name return count(p)"
    res = session.run(cql,name=entity)
    ans = 0
    for record in res:
        ans = record.values()[0]
    return ans

def GetRelationNumsig(entity):
    '''根据实体名，得到一跳内的不同关系个数,重要度'''
    cql= "match p=(a:Entity)-[r1:Relation]-() where a.name=$name return DISTINCT r1.name"
    rpath = []
    res = session.run(cql,name=entity)
    for record in res:
        rpath.append(record['r1.name'])
    return len(set(rpath))
                      
def GetTwoEntityTuple(e1,r1,e2):
    cql = "match (a:Entity)-[r1:Relation]-(b:Entity)-[r2:Relation]-(c:Entity) where a.name=$e1n and r1.name=$r1n and c.name=$e2n return DISTINCT r2.name"
    tuples = []
    res = session.run(cql,e1n=e1,r1n=r1,e2n=e2)
    for record in res:
        tuples.append(tuple([e1,r1,record['r2.name'],e2]))
    return tuples


if __name__ == '__main__':
    
    sql = "match (a:Entity)-[r1:Relation]-(c)-[r2:Relation]-(b:Entity) where a.name=$ename1 and r1.name=$rname1 and r2.name=$rname2 and b.name=$ename2 return c.name"
    res = session.run(sql,ename1='<维力医疗>',rname1='<公司高管>',rname2='<硕士学位>',ename2='<高分子材料专业>')
    ans = [record['c.name'] for record in res]
    print(ans)