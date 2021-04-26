import torch
import utils
import time
import numpy as np
docs1 = open('docs.txt','r',encoding='utf-8').readlines()
docs2 = open('docs_no_service.txt','r',encoding='utf-8').readlines()
docs = [{'doc':i[:-1],'label':1} for i in docs1]
docs += [{'doc':i[:-1],'label':2} for i in docs2]
keywords = open('keywords.txt','r',encoding='utf-8').readlines()
titles1 = open('titles.txt','r',encoding='utf-8').readlines()
titles2 = open('titles_no_service.txt','r',encoding='utf-8').readlines()
titles = [{'doc':i[:-1],'label':1}for i in titles1]
titles += [{'doc':i[:-1],'label':2}for i in titles2]
keywords = [line.split('|')for line in keywords]
W2V = utils.getW2V()
print(keywords)
def keywords_docs(keywords,docs,outname,flag):
    '''

    :param keywords:
    :param docs:
    :param outname:
    :param flag: 1 means use keywords with abstract,other not use
    :return:
    '''
    times = [0.0 for i in range(10)]
    tmp_time = [0.0 for i in range(10)]
    out = open(outname, 'w', encoding='utf-8')
    for key,abstract in keywords:
        print(key)
        tmp = key


        if(flag == 1):tmp = ''.join([key,abstract[:-1]])
        scores = []
        key_onegram = utils.one_gram(tmp)
        key_jieba = utils.fenci(tmp)
        label = ''
        for item in docs:
            doc = item['doc']
            label = item['label']
            tmp_time[0] = time.time()
            doc_jieba = utils.fenci(doc[:-1])
            doc_onegram = utils.one_gram(doc[:-1])
            tmp_time[1] = time.time()
            TF = utils.get_TF(doc[:-1],key_onegram)
            tmp_time[2] = time.time()
            jaccard = utils.jaccard_common_words(tmp,doc[:-1])
            tmp_time[3] = time.time()
            levenshtein = utils.Levenshtein_distance(tmp,doc[:-1])
            tmp_time[4] = time.time()
            ochiai = utils.ochiai_common_words(tmp,doc[:-1])
            tmp_time[5] = time.time()
            W2V_AVG_sim_onegram = utils.get_avg_sim(key_onegram,doc_onegram,W2V)
            tmp_time[6] = time.time()
            W2V_AVG_sim_jieba = utils.get_avg_sim(key_jieba, doc_jieba, W2V)
            ans = TF + jaccard + ochiai + 0.5*W2V_AVG_sim_onegram + 0.5*W2V_AVG_sim_jieba
            #print(ans)
            #print(levenshtein)
            #print(TF,jaccard,ochiai,W2V_AVG_sim_jieba)
            for i in range(6):
                times[i] += tmp_time[i+1]-tmp_time[i]
            scores.append({'score': ans, 'text': doc[:100 if len(doc)>100 else len(doc)], 'label': label})
        scores = sorted(scores, key=lambda x: x['score'], reverse=True)

        s = [str(key), '\n']
        pred_list = [0.0 for i in range(len(docs))]
        cnt = 0
        for d in scores:
            for k, value in d.items():
                if (k == 'text'): continue
                if k == 'label':
                    pred_list[cnt] = 2 - value
                    cnt += 1
                s.append(str(k))
                s.append(':')
                s.append(str(value))
                s.append('\t')
            s+=['text', ':', d['text']]
            s.append('\n')
        rank_list = sorted(pred_list,reverse=True)
        print(rank_list)
        print(pred_list)
        ndcg = utils.getNDCG(np.array(rank_list),np.array(pred_list))

        out.write(''.join(s))
        out.write('ndcg:'+str(ndcg)+'\n')
    print(times)

keywords_docs(keywords,titles,'keywords_titles.txt',flag = 0)
keywords_docs(keywords,titles,'keywords_titles_with_abstract.txt',flag = 1)
keywords_docs(keywords,docs,'keywords_docs.txt',flag = 0)
keywords_docs(keywords,docs,'keywords_docs_with_abstract.txt',flag = 1)



