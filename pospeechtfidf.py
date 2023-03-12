# coding=utf-8
import sys
import math
import numpy as np
import jieba
from jieba import analyse as anal
import codecs
import jieba.posseg as pos
import json
from decimal import Decimal
#计算idf值
cshuru=open('data/cuttraindocs.txt','r',encoding='utf-8')
cutdocs=[]
for i in cshuru:
    cutdocs.append(i.strip())
cutdocs=[x for x in cutdocs if x !='']
def trainidf(textdate):
    idfw={}
    N=len(textdate)
    for i in textdate:
        for w in set(i.split()):
            idfw[w]=idfw.get(w,0.0)+1.0
    for x,y in idfw.items():
        idfw[x]=math.log(N/(1.0+y))
    return idfw
#计算tf
def traintf(onetext):
    tfw={}
    sonetext=onetext.split()
    N=len(sonetext)
    for i in sonetext:
        tfw[i]=tfw.get(i,0.0)+1.0
    for i,j in tfw.items():
        tfw[i]='%.6f'%(j/N)
    return tfw#tfw是字典
#求词性的权值
def posw(word):
    cc = pos.cut(word)
    c=0
    for word, flag in cc:
        if flag[0]=='n':
            c=0.69
        elif flag[0]=='v':
            c=0.11
        else:
            c=0.2
    return c
#计算tfidf
def traintfidf(onetext,idfw,l):
    tfw=traintf(onetext)
    tfidfw={}
    c=0
    for x,y in tfw.items():
        c=posw(x)*l
        tfidfw[x]=float(Decimal(str(idfw.get(x, 0)))*Decimal(str(y))*Decimal(str(c)))#Decimal是用来十进数运算，没有这个，1.1+2.2不等于3.3
    return tfidfw#tfidfw是字典
def alltfidf(cutdocs,l):
    idf = trainidf(cutdocs)
    tfidf = []
    trainkeyword = []
    for i in range(len(cutdocs)):
        tfidfw = traintfidf(cutdocs[i], idf, l)
        tfidf.append(tfidfw)
        tkeyword = sorted(tfidfw.items(), key=lambda x: x[1], reverse=True)
        trainkeyword.append(tkeyword)
    return trainkeyword
#trainkeyword=alltfidf(cutdocs,1)
#tfidf是由字典构成的列表，trainkeyword是由元组构成的列表构成的列表。
#保存计算的trainkeyword数据
'''
f=open('data/traintfidf.txt','w',encoding='utf-8')
for i in trainkeyword:
    for j in range(len(i)):
        f.write(i[j][0])
        f.write(':')
        f.write(str(i[j][1]))
        f.write('\t')
    f.write('\n')
f.close()
'''
keywordlist1=open('data/train_docs_keywords.txt','r',encoding='utf-8')
keywordlist=[]
for j in keywordlist1:
    s = j.split()[1]
    keywordlist.append(s.replace(',',' ').split())
#print(keywordlist)#keywordlist是由列表构成的列表
def aprf(text1,text2):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(text1)):
        for j in range(len(text2[i])):
            if j < 3:
                if text2[i][j][0] in text1[i]:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if text2[i][j][0] in text1[i]:
                    fn = fn + 1
                else:
                    tn = tn + 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy,precision,recall,f1,tp,tn,fp,fn
if __name__ == '__main__':#方便其他程序调用
    for i in range(0,21):
        trainkeyword = alltfidf(cutdocs, i*0.01)
        a, p, r, f, tp, tn, fp, fn = aprf(keywordlist, trainkeyword)
        print(a, p, r, f, tp, tn, fp, fn)