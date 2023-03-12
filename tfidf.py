import sys
import math
import numpy as np
import jieba
from jieba import analyse as anal
import codecs
import json
from decimal import Decimal
'''
shuru=open('data/alldocs1.txt','r',encoding='utf-8')
#alldocs1是去掉空格之后的文本
#all_docs是原文本
#alldocs是只有十个例子的文本
#shuru=open('D:/deeplearning/py-project/Keyword-Extraction-master/data/test_trg.txt','r',encoding='utf-8')
#不加encoding='utf-8'会报错
keywordlist1=open('data/train_docs_keywords.txt','r',encoding='utf-8')
#下面读取已知关键词对应的序号
keywordzuobiao=[]
for i in keywordlist1:
    keywordzuobiao.append(i.strip('D').split()[0])
#print(keywordzuobiao)
#必须要重新打开，下面的才能运行，不然keywordlist1就是空的
#读取关键词
keywordlist1=open('data/train_docs_keywords.txt','r',encoding='utf-8')
keywordlist=[]
for j in keywordlist1:
    s = j.split()[1]
    keywordlist.append(s.replace(',',' ').split())
#print(keywordlist)

#句子提取
#处理数据，将原文与给出的关键词对应
jvzi=[]
for i in shuru:
    z=i.split()
    z.pop(0)
    jvzi.append(z)
#加上codecs后，写入才不会出现unicodeencodeerror
f=codecs.open('data/traindocs.txt','w','utf-8')
for i in keywordzuobiao:
    k=int(i)
    for j in range(0,len(jvzi[k-1])):
        #要记得列表的指标是从零开始
        f.writelines(jvzi[k-1][j])
        f.write(' ')#"\t"是一个缩进，空格直接用' '
    f.write("\n")
f.close()
'''
'''
#停用词表
stopwordlist1=open('data/hit_stopwords.txt',encoding='utf-8').readlines()
stopwordlist=[]
for i in stopwordlist1:
    stopwordlist.append(i.replace('\n',''))
#print(stopwordlist)
#去除停用词
xshuru=open('data/traindocs.txt','r',encoding='utf-8')
cutshuru1=[]
for i in xshuru:
    c=list(jieba.cut(i))#将分词后的句子变为列表
    while ' ' in c:
        c.remove(' ')
    cutshuru1.append(c)#要注意空格
f=codecs.open('data/cuttraindocs.txt','w','utf-8')
cutshuru=[]
for i in range(0,len(cutshuru1)):
    cc=[]
    for j in range(0,len(cutshuru1[i])):
        if not cutshuru1[i][j] in stopwordlist:
            cc.append(cutshuru1[i][j])
            f.writelines(cutshuru1[i][j])#将分完词并去除停用词的文本保存
            f.write(' ')
    #f.write("\n")
    cutshuru.append(cc)#去除停用词
f.close()
#print(cutshuru)
'''

#计算idf值
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
#计算tfidf
def traintfidf(onetext,idfw):
    tfw=traintf(onetext)
    tfidfw={}
    for x,y in tfw.items():
        tfidfw[x]=float(Decimal(str(idfw[x]))*Decimal(str(y)))#Decimal是用来十进数运算，没有这个，1.1+2.2不等于3.3
    return tfidfw#tfidfw是字典
idf=trainidf(cutdocs)
tfidf=[]
trainkeyword=[]
for i in range(len(cutdocs)):
    tfidfw=traintfidf(cutdocs[i],idf)
    tfidf.append(tfidfw)
    tkeyword=sorted(tfidfw.items(),key=lambda x:x[1],reverse=True)
    trainkeyword.append(tkeyword)
#tfidf是由字典构成的列表，trainkeyword是由元组构成的列表构成的列表。
#保存计算的trainkeyword数据
f=open('date/traintfidf.txt','w',encoding='utf-8')
for i in trainkeyword:
    for j in range(len(i)):
        f.write(i[j][0])
        f.write(':')
        f.write(str(i[j][1]))
        f.write('\t')
    f.write('\n')
f.close()
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
    a, p, r, f,tp,tn,fp,fn = aprf(keywordlist, trainkeyword)
    print(a, p, r, f,tp,tn,fp,fn)
