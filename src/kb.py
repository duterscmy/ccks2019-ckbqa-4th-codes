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
endtime = time.time()
print ('start neo4j and match all entities,the time is %.2f'%(endtime-begintime))

def GetRelationPaths(entity):
    '''根据实体名，得到所有1跳2跳关系list,2d-list'''

    cql_1 = "match (a:Entity)-[r1:Relation]-() where a.name=$name return DISTINCT r1.name"
    cql_2 = "match (a:Entity)-[r1:Relation]->()-[r2:Relation]->() where a.name=$name return DISTINCT r1.name,r2.name"
    #cql_2 = "match (a:Entity)-[r1:Relation]-()-[r2:Relation]->() where a.name=$name return DISTINCT r1.name,r2.name"
    rpaths1 = []
    res = session.run(cql_1,name=entity)
    for record in res:
        rpaths1.append([record['r1.name']])
    rpaths2 = []
    res = session.run(cql_2,name=entity)
    for record in res:
        rpaths2.append([record['r1.name'],record['r2.name']])
    return rpaths1+rpaths2

def GetRelations_2hop(entity):
    cql_2 = "match (a:Entity)-[r1:Relation]-()-[r2:Relation]->() where a.name=$name return DISTINCT r1.name,r2.name"
    rpaths2 = []
    res = session.run(cql_2,name=entity)
    for record in res:
        rpaths2.append([record['r1.name'],record['r2.name']])
    dic = {}
    for rpath in rpaths2:
        for r in rpath:
            dic[r] = 0
    return dic

def GetRelationPath_BetweenEntitys(e1,e2):
    '''
    给定两个实体，返回二者间的关系序列列表
    '''
    cql_2 = "match (a:Entity)-[r1:Relation]-()-[r2:Relation]-(b:Entity) where a.name=$name and b.name=$name2 return DISTINCT r1.name,r2.name"
    rpaths = []
    res = session.run(cql_2,name=e1,name2=e2)
    for record in res:
        rpaths.append([record['r1.name'],record['r2.name']])
    return rpaths

def Get_Tuples_class1(entity):
    '''根据实体名，得到所有1跳'''

    cql_1 = "match (a:Entity)-[r1:Relation]-() where a.name=$name return DISTINCT r1.name"
    rpaths = []
    res = session.run(cql_1,name=entity)
    for record in res:
        rpaths.append([record['r1.name']])
    return rpaths

def Get_Tuples_class2(entity):
    '''根据实体名，得到固定逻辑形式的2跳关系list,2d-list'''

    cql_2 = "match (a:Entity)-[r1:Relation]->(b:Entity)-[r2:Relation]->() where a.name=$name return DISTINCT r1.name,r2.name"
    rpaths = []
    res = session.run(cql_2,name=entity)
    for record in res:
        rpaths.append([record['r1.name'],record['r2.name']])
    return rpaths

def Get_Tuples_class3(entity):
    '''根据实体名，得到固定逻辑形式的2跳关系list,2d-list'''

    cql_2 = "match (a:Entity)<-[r1:Relation]-(b:Entity)-[r2:Relation]->() where a.name=$name return DISTINCT r1.name,r2.name"
    rpaths = []
    res = session.run(cql_2,name=entity)
    for record in res:
        rpaths.append([record['r1.name'],record['r2.name']])
    return rpaths

def Get_Tuples_class4(entity):
    '''根据实体名，得到固定逻辑形式的2跳关系list,2d-list'''

    cql_2 = "match (a:Entity)-[r1:Relation]->(b:Entity)<-[r2:Relation]-() where a.name=$name return DISTINCT r1.name,r2.name"
    rpaths = []
    res = session.run(cql_2,name=entity)
    for record in res:
        rpaths.append([record['r1.name'],record['r2.name']])
    return rpaths

def GetTwoEntityTuple(e1,r1,e2):
    cql = "match (a:Entity)-[r1:Relation]-(b:Entity)-[r2:Relation]-(c:Entity) where a.name=$e1n and r1.name=$r1n and c.name=$e2n return DISTINCT r2.name"
    tuples = []
    res = session.run(cql,e1n=e1,r1n=r1,e2n=e2)
    for record in res:
        tuples.append(tuple([e1,r1,record['r2.name'],e2]))
    return tuples


if __name__ == '__main__':
    def runcql3(tx,cql,ename1,ename2,rname1,rname2):#用于查询实体直接相连的关系
        l = []
        for record in tx.run(cql,name1=ename1,name2=ename2,name3=rname1,name4=rname2):
            l.append([record['b.name']])
        return l
    cql = "match (a:Entity)-[r1:Relation]-(b:Entity)-[r2:Relation]-(c:Entity) where a.name=$name1 and c.name=$name2 and r1.name=$name3 and r2.name=$name4 return b.name"
    res = session.run(cql,name1='<复旦大学>',name2='<政治家>',name3='<毕业院校>',name4='<职业>')
    l = [record['b.name'] for record in res]
    print (l)
    session.close()