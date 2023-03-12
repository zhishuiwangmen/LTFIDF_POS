import tfidf as ti
import multfactfidf as mfti
import ngramtfidf as nti
import jieba
import json
from decimal import Decimal
def judge(trainkeyngram,trainkey):
    tkng=dict(trainkey)
    for i in range(3):#这里可以变
        m=0
        p=[]
        q={}
        t=[]
        c=list(jieba.cut(trainkeyngram[i][0]))
        for j in range(len(c)):
            for k in range(len(trainkey)):#这里可以变
                if c[j]==trainkey[k][0]:
                    m=m+1
                    p.append(trainkey[k][1])
                    t.append(trainkey[k][0])
        if m==len(c):
            r=0
            for s in p:
                r=Decimal(str(r))+Decimal(str(s))
            r=float(Decimal(r)+Decimal(trainkeyngram[i][1]))
            q[trainkeyngram[i][0]]=r
            for b in t:#分完词之后，有的会重合
                if b in tkng:
                    del tkng[b]
            tkng.update(q)
    mtkng = sorted(tkng.items(), key=lambda x: x[1], reverse=True)
    return mtkng
def multngram(trainkey,trainkeyngram):
    multngram=[]
    for i in range(len(trainkey)):
        x=judge(trainkeyngram[i],trainkey[i])
        multngram.append(x)
    return multngram
keywordlist1=open('date/train_docs_keywords.txt','r',encoding='utf-8')
keywordlist=[]
for j in keywordlist1:
    s = j.split()[1]
    keywordlist.append(s.replace(',',' ').split())
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
if __name__ == '__main__':
    trainkeyngram = nti.ngramkey(3,1)
    #trainkey = ti.trainkeyword
    trainkey = mfti.trainkeyword
    mngram=multngram(trainkey,trainkeyngram)
    a, p, r, f,tp,tn,fp,fn = aprf(keywordlist, mngram)
    print(a, p, r, f,tp,tn,fp,fn)

