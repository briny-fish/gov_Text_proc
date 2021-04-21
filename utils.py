import gensim
import os
import Levenshtein
import numpy as np
import pickle
import math
import jieba
from gensim.models import KeyedVectors
#model = KeyedVectors.load_word2vec_format('sgns.renmin.bigram-char.bz2',binary = False, encoding = "utf-8", unicode_errors = "ignore")
def getW2V():
    word2vec = open('w2v.pkl','rb')
    model = pickle.load(word2vec)
    return model
def get_IDF():
    '''

    :param file:
    :return:
    '''
def Levenshtein_distance(text1, text2):
    '''

    :param text1: a list of terms
    :param text2:
    :return:
    '''
    text1 = ''.join(text1)
    text2 = ''.join(text2)
    return Levenshtein.distance(text1, text2)

def jaccard_common_words(text1, text2):
    '''

    :param text1: a list of terms
    :param text2:
    :return:
    '''
    str1 = set(text1)
    str2 = set(text2)
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    return float(len(str1 & str2)) / len(str1 | str2)

def ochiai_common_words(text1, text2):
    '''

    :param text1: a list of terms
    :param text2:
    :return:
    '''
    str1 = set(text1)
    str2 = set(text2)
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    return float(len(str1 & str2)) / math.sqrt(len(str1) * len(str2))

def cosin(vecA,vecB):
    vecA = np.array(vecA)
    vecB = np.array(vecB)
    return vecA.dot(vecB) / np.linalg.norm(vecA) / np.linalg.norm(vecB)

def get_avg_sim(texta,textb,W2V):
    '''

    :param texta: a list of terms
    :param textb: a list of terms
    :param W2V:
    :return: a float score
    '''
    vocabs = W2V.vocab.keys()
    vecA = None
    vecB = None
    lenA = 0
    lenB = 0
    for term in texta:
        if term in vocabs:
            vecA += W2V[W2V.vocab[term].index]
            lenA += 1
    for term in textb:
        if term in vocabs:
            vecB += W2V[W2V.vocab[term].index]
            lenB += 1

    vecA = vecA / float(lenA)
    vecB = vecB / float(lenB)
    return cosin(vecA,vecB)
def get_TF(doc,term):
    '''

    :param doc: a list of terms
    :param term: a term(1,2,3 or other grams)
    :return:
    '''
    doc = ''.join(doc)
    count = float(len(doc))
    tf = 0
    for i in range(len(doc)):
        if(i+len(term)>len(doc)):break
        if(doc[i:i+len(term)]==term):tf+=1
    return float(tf)/count
'''
txt = open("text_remove.txt", "r", encoding='utf-8').read()
words = jieba.lcut(txt)     # 使用精确模式对文本进行分词
counts = {}     # 通过键值对的形式存储词语及其出现的次数

for word in words:
    if len(word) == 1:    # 单个词语不计算在内
        continue
    else:
        counts[word] = counts.get(word, 0) + 1    # 遍历所有词语，每出现一次其对应的值加 1

items = list(counts.items())
items.sort(key=lambda x: x[1], reverse=True)    # 根据词语出现的次数进行从大到小排序

for i in range(3):
    word, count = items[i]
    print("{0:<5}{1:>5}".format(word, count))
'''
print(get_TF(['我','是你','是我'],'是'))
print(cosin([1.0,0.5],[-1.0,0.5]))