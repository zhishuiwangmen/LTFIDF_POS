# coding=utf-8
import sys
import math
import numpy as np
import jieba
from jieba import analyse as anal
import codecs
import json
from decimal import Decimal
def ngramkey(n,step):
    def ngram(text, n,step):
        aa = '，。、《》？！@#￥%……&*（）——+{}【】|：；‘’“”=-~<>,./?'';:""{}|[]\!$^()_'
        bb = '                                                        '
        trans = text.maketrans(aa, bb)
        cc = text.translate(trans)
        ccc = cc.strip(' ').split()
        cccc = []
        for i in ccc:
            if len(i) > n:
                for j in range(0, len(i) - n + 1, step):
                    k = i[j:j + n]
                    cccc.append(k)
                if (len(i) - n) % step != 0:#判断语句是否有未分割字符
                    kk = i[(len(i) - n):len(i)]
                    cccc.append(kk)
            else:
                cccc.append(i)
        return (cccc)  # 由字符串组成的列表

    xshuru = open('data/traindocs.txt', 'r', encoding='utf-8')
    cutshuru1 = []
    for i in xshuru:
        c = ngram(i, n,step)
        cutshuru1.append(c)  # cutshuru1由字符串构成的列表构成的列表
    # 保存计算的ngram分词
    f = open('data/ngramcut.txt', 'w', encoding='utf-8')
    for i in cutshuru1:
        for j in range(len(i)):
            f.write(i[j])
            f.write('\t')
        f.write('\n')
    f.close()
    def trainidf(textdate):
        idfw = {}
        N = len(textdate)
        for i in textdate:
            for w in set(i):  # set创建无序不重复元素集，注意要将i转化成列表，才能得到关于词的不重复集
                idfw[w] = idfw.get(w, 0.0) + 1.0  # 如果不加0.0，会报错，开始循环时无法赋值
        for x, y in idfw.items():  # items遍历字典列表
            idfw[x] = math.log(N / (1.0 + y))  # 6.214608098422191是只有一篇文章有对应词
        return idfw

    # 计算tf
    def traintf(sonetext):
        tfw = {}
        N = len(sonetext)
        for i in sonetext:
            tfw[i] = tfw.get(i, 0.0) + 1.0
        for i, j in tfw.items():
            tfw[i] = '%.6f' % (j / N)
        return tfw  # tfw是字典

    # 计算tfidf
    def traintfidf(onetext, idfw):
        tfw = traintf(onetext)
        tfidfw = {}
        for x, y in tfw.items():
            tfidfw[x] = float(Decimal(str(idfw.get(x, 0))) * Decimal(str(y)))  # Decimal是用来十进数运算，没有这个，1.1+2.2不等于3.3
        return tfidfw  # tfidfw是字典

    idf = trainidf(cutshuru1)
    tfidf = []
    trainkeyword = []
    for i in range(len(cutshuru1)):
        tfidfw = traintfidf(cutshuru1[i], idf)
        tfidf.append(tfidfw)
        tkeyword = sorted(tfidfw.items(), key=lambda x: x[1], reverse=True)
        trainkeyword.append(tkeyword)
    return trainkeyword
trainkeyword=ngramkey(3,1)
#tfidf是由字典构成的列表，trainkeyword是由元组构成的列表构成的列表。
#保存计算的trainkeyword数据

f=open('data/ngramtraintfidf.txt','w',encoding='utf-8')
for i in trainkeyword:
    for j in range(0,3):
        f.write(i[j][0])
        f.write(':')
        f.write(str(i[j][1]))
        f.write('\t')
    f.write('\n')
f.close()

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
            if j < 3:#表示关键词的取值范围
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