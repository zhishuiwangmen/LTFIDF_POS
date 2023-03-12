import sys
import math
import numpy as np
import jieba
from jieba import analyse as anal
import codecs
import json
from decimal import Decimal
xshuru=open('date/traindocs.txt','r',encoding='utf-8')
x=xshuru.readlines()
#计算idf值
cshuru=open('date/cuttraindocs.txt','r',encoding='utf-8')
cutdocs=[]
for i in cshuru:
    cutdocs.append(i.strip())
cutdocs=[x for x in cutdocs if x !='']#去除文本中的空字符
#print(cutdocs[0])
#计算词语的整体离散度
'''
def indword(cuttext):
    ind={}
    ctext=' '.join(cuttext)
    for i in set(ctext.strip().split()):
        w=[]
        for j in range(len(cuttext)):
            if i in cuttext[j].strip().split():
                #ind[i]=j
                w.append(j)
        ind[i]=w
    print(len(ind))
    return ind

def onetf(onetext, t):
    l = onetext.count(t)
    if len(onetext) == 0:
        return 0
    else:
        ll = l / len(onetext)
        return ll
def alldis(text,cuttext):
    aa = '，、《》？！@#￥……&*（）——+{}【】|：；‘’“”=~-<>?":;。\/|,[]_^$!`''()'
    bb = '                                                    '
    alltext = []
    dis={}
    for i in range(len(text)):
        lt=[]
        trans = text[i].maketrans(aa, bb)
        s = text[i].translate(trans)
        k=jieba.cut(s)
        for j in k:
            lt.append(j)
        alltext.append(lt)
    print(1)
    iw=indword(cuttext)

    for word,ind in iw.items():
        stf = []
        sumtf = 0
        avetf = 0
        d=0
        di = 0
        for i in ind:
            tf=onetf(alltext[i],word)
            stf.append(tf)
            sumtf=sumtf+tf
        avetf=sumtf/len(ind)
        for i in stf:
            d=d+(i-avetf)**2
        d=(d/len(ind))**0.5
        di=d/avetf
        dis[word]=di
    return dis
dis=alldis(x,cutdocs)
np.save('./date/alldisvalue.npy',dis)
'''
dis=np.load('date/alldisvalue.npy',allow_pickle=True).item()
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
#计算tfidf
def traintfidf(onetext,idfw,dis,k):
    tfw=traintf(onetext)
    tfidfw={}
    for x,y in tfw.items():
        tfidfw[x]=float(Decimal(str(idfw[x]))*Decimal(str(y))+Decimal(str(k*dis[x])))#Decimal是用来十进数运算，没有这个，1.1+2.2不等于3.3
    return tfidfw#tfidfw是字典
def distfidf(cutdocs,dis,k):
    idf = trainidf(cutdocs)
    tfidf = []
    trainkeyword = []
    for i in range(len(cutdocs)):
        tfidfw = traintfidf(cutdocs[i], idf, dis,k)
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
if __name__ == '__main__':#方便其他程序调用
    for i in range(0, 100):
        k=round(i*0.000001,5)
        trainkeyword = distfidf(cutdocs,dis,k)
        a, p, r, f, tp, tn, fp, fn = aprf(keywordlist, trainkeyword)
        print(a, p, r, f, tp, tn, fp, fn,k)