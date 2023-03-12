import sys
import math
import numpy as np
import jieba
from jieba import analyse as anal
import jieba.posseg as pos
import codecs
import json
from decimal import Decimal
#计算idf值
xshuru=open('date/traindocs.txt','r',encoding='utf-8')
x=xshuru.readlines()
cshuru=open('date/cuttraindocs.txt','r',encoding='utf-8')
cutdocs=[]
for i in cshuru:
    cutdocs.append(i.strip())
cutdocs=[x for x in cutdocs if x !='']#去除文本中的空字符
#print(cutdocs[0])
def trainidf(textdate):
    idfw={}
    N=len(textdate)
    for i in textdate:
        for w in set(i.split()):#set创建无序不重复元素集，注意要将i转化成列表，才能得到关于词的不重复集
            idfw[w]=idfw.get(w,0.0)+1.0#如果不加0.0，会报错，开始循环时无法赋值
    for x,y in idfw.items():#items遍历字典列表
        idfw[x]=math.log(N/(1.0+y))#6.214608098422191是只有一篇文章有对应词
    return idfw
#idf=trainidf(cutdocs)
#print(idf)
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
#计算词语索引
def posit(text):
    kk={}
    for i in set(text):
        k=[]
        for item, value in enumerate(text):
            if value == i:
                k.append(item)
        kk[i]=k#键是词语，值是列表
    return kk
#正态分布函数
def nordis(x,mu,delta):
    f=(math.exp(-((x-mu)**2)/2*(delta**2)))/((2*math.pi)**0.5*delta)
    return f
def positw(text,lam):
    t=posit(text)
    l=len(text)/2
    w={}
    for word in t:
        v=0
        ind=t[word]
        if len(ind)!= 1:
            v1=(ind[0]/l-1)*lam
            v2=(ind[len(ind)-1]/l-1)*lam
            v=1-(nordis(v1,0,0.399)+nordis(v2,0,0.399))/2
            w[word]=v
        else:
            v3=(ind[0]/l-1)*lam
            v=1-nordis(v3,0,0.399)
            w[word] = v
    return w
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
def traintfidf(onetext,idfw,lam,l):
    tfw=traintf(onetext)
    w=positw(onetext.split(),lam)#onetext是字符串，需要转化成列表
    tfidfw={}
    c = 0
    for x,y in tfw.items():
        z=w[x]
        c = posw(x) * l
        tfidfw[x]=float(Decimal(str(idfw.get(x, 0)))*Decimal(str(y))*Decimal(str(z))*100+Decimal(str(c)))#Decimal是用来十进数运算，没有这个，1.1+2.2不等于3.3
    return tfidfw#tfidfw是字典
def multtfidf(cutdocs,lam,l):
    idf = trainidf(cutdocs)
    tfidf = []
    trainkeyword = []
    for i in range(len(cutdocs)):
        tfidfw = traintfidf(cutdocs[i],idf,lam,l)
        tfidf.append(tfidfw)
        tkeyword = sorted(tfidfw.items(), key=lambda x: x[1], reverse=True)
        trainkeyword.append(tkeyword)
    return trainkeyword
#tfidf是由字典构成的列表，trainkeyword是由元组构成的列表构成的列表。

keywordlist1=open('date/train_docs_keywords.txt','r',encoding='utf-8')
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
trainkeyword=multtfidf(cutdocs,0.42,0.14)
if __name__ == '__main__':#方便其他程序调用
    '''
    for i in range(0, 21):
        lam=round(i*0.01+0.3,5)
        trainkeyword = multtfidf(cutdocs,lam,0.14)
        a, p, r, f, tp, tn, fp, fn = aprf(keywordlist, trainkeyword)
        print(a, p, r, f, tp, tn, fp, fn,lam)'''
    a, p, r, f, tp, tn, fp, fn = aprf(keywordlist, trainkeyword)
    print(a, p, r, f, tp, tn, fp, fn)