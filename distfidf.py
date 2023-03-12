import sys
import math
import numpy as np
import jieba
from jieba import analyse as anal
import codecs
import json
from decimal import Decimal

xshuru=open('D:/BaiduNetdiskDownload/traindocs.txt','r',encoding='utf-8')
x=xshuru.readlines()
#计算idf值
cshuru=open('D:/BaiduNetdiskDownload/cuttraindocs.txt','r',encoding='utf-8')
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
#计算文内离散度
def sentf(sentence,t):
    l=sentence.count(t)
    #print(type(l))
    if len(sentence) == 0:
        return 0
    else:
        ll = l / len(sentence)
        return ll

def dissentf(text,cuttext):#这里的text是由句子组成的列表
    c = set(cuttext.split())
    x = text.replace('。', ' ')
    xx = x.strip().split()
    dis = {}
    cutsen = []
    aa = '，、《》？！@#￥……&*（）——+{}【】|：；‘’“”=~-<>?":;\/|,[]_^$!`''()'
    bb = '                                                   '
    for i in range(len(xx)):
        trans = xx[i].maketrans(aa, bb)
        s = xx[i].translate(trans)
        ss = s.strip().split()
        sss = []
        for m in ss:
            k = jieba.cut(m)
            for j in k:
                sss.append(j)
        cutsen.append(sss)
    for i in c:
        avetf = 0
        sumtf = 0
        m = 0
        stf = []
        d = 0
        di = 0
        p = []
        for j in range(len(cutsen)):
            tf = sentf(cutsen[j], i)
            stf.append(tf)
            sumtf = sumtf + tf
        avetf = sumtf / len(cutsen)
        for k in stf:
            d = d + (k - avetf) ** 2
        d=d+(len(cutsen)-len(stf))*(avetf**2)
        d = (d / (len(cutsen)-1))**0.5
        di = d/avetf
        dis[i] = di
    return dis
#计算tfidf
def traintfidf(text,onetext,idfw,k):
    tfw=traintf(onetext)
    dis = dissentf(text, onetext)
    tfidfw={}
    for x,y in tfw.items():
        tfidfw[x]=float(Decimal(str(idfw[x]))*Decimal(str(y))+Decimal(str(k-k*dis[x])))#Decimal是用来十进数运算，没有这个，1.1+2.2不等于3.3
    return tfidfw#tfidfw是字典
def distfidf(x,cutdocs,k):
    idf = trainidf(cutdocs)
    tfidf = []
    trainkeyword = []
    for i in range(len(cutdocs)):
        tfidfw = traintfidf(x[i], cutdocs[i], idf, k)
        tfidf.append(tfidfw)
        tkeyword = sorted(tfidfw.items(), key=lambda x: x[1], reverse=True)
        trainkeyword.append(tkeyword)
    return trainkeyword

#tfidf是由字典构成的列表，trainkeyword是由元组构成的列表构成的列表。

keywordlist1=open('D:/BaiduNetdiskDownload/train_docs_keywords.txt','r',encoding='utf-8')
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
    for i in range(0, 20):
        k=round(i*0.1,5)
        trainkeyword = distfidf(x, cutdocs, k)
        a, p, r, f, tp, tn, fp, fn = aprf(keywordlist, trainkeyword)
        print(a, p, r, f, tp, tn, fp, fn,k)