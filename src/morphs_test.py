# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 01:06:22 2020

@author: user
"""


#%%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import re
import time
from collections import Counter
from wordcloud import WordCloud
from konlpy.tag import Twitter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import konlpy 
from konlpy.tag import Okt 


#%%
# load file
xlxs_dir = '런닝맨 - 스브스 공식 채널_video_info.xlsx'
df_read_video_info = pd.read_excel(xlxs_dir, sheet_name = 'video')
df_read_comment = pd.read_excel(xlxs_dir, sheet_name = 'comment')
#%%
# 긍정 단어 리스트
f = open('positive_words_self.txt', 'r',encoding='UTF-8')
lines = f.readlines()
f.close()

positive=[]
for i in lines:
    i=i.replace('\n','')
    positive.append(i)
   
okt=Okt()

positive_word=[]
for sentence in positive:
    temp_X = okt.morphs(str(sentence), stem=True) 
    positive_word.append(temp_X[:])


#%%
# 부정 단어 리스트
f = open('negative_words_self.txt', 'r',encoding='UTF-8')
lines = f.readlines()
f.close()

negative=[]
for i in lines:
    i=i.replace('\n','')
    negative.append(i)
    
#%%
# 이모티콘 제거
emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u'\U00010000-\U0010ffff'  # not BMP characters
    "]+", flags=re.UNICODE)

# 분석에 어긋나는 불용어구 제외 (특수문자, 의성어)
han = re.compile(r'[ㄱ-ㅎㅏ-ㅣ!?~,".\n\r#\ufeff\u200d]')
 
# 그 다음으로는 기존의 데이터에서 댓글컬럼만 뽑아냅니다
comment_list = []
for i in range(len(df_read_comment)):
    comment_list.append(df_read_comment['comment'].iloc[i])
 

# 최종적으로 compile한 문자열을 이용하여 불용어구를 제외하고 댓글을 보기 쉽게 데이터 프레임으로 저장합니다.
comment_result = []
for i in comment_list:
    tokens = re.sub(emoji_pattern,"",str(i))
    tokens = re.sub(han,"",tokens)
    comment_result.append(tokens)

df_comment_result = pd.DataFrame(comment_result, columns=["comment"])

#%% 불용어 제거

# 불용어리스트 로드
f = open('불용어리스트.txt', 'r',encoding='UTF-8')
lines = f.readlines()
f.close()
stop_words=[]
for i in lines:
    stop_words.append(i.replace('\n',''))
print(stop_words)

# 토큰화
words=[]
for i in range(len(df_comment_result)):
    word_tokens=[]
    word_tokens = word_tokenize(str(df_comment_result['comment'].iloc[i]))
    print(word_tokens)
    words.append(word_tokens)

# 불용어 제거
words2=[]
for i in words:
    a=[]
    for z in i:
        if z not in stop_words:
            a.append(z)
    words2.append(a)

#%% 형태소 추출
    

words4=[]
for i in words2:
    c=[]
    for z in i:
        b=okt.morphs(str(z),stem=True)
        c.append(' '.join(b))
    words4.append(' '.join(c).split())

words2=words4
#%% 

#1. comment의 단어가 positive,negative 단어 리스트에 있는지 확인

label=[]    

for i in words2:
    pos=0
    neg=0
    for z in i:
        
        if z in positive:
            pos+=1
        elif z in negative:
            neg+=1
    if pos>neg:
        label.append('1')
    elif pos<neg:
        label.append('-1')
    else:
        label.append('0')

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
#2. 유사도 측정

tfidf_vectorizer=TfidfVectorizer()
words

tfidf_vectorizer = TfidfVectorizer(min_df=1)

label2=[]
for i in words2:
    a=[]
    a.append(' '.join(i))
    a.append(' '.join(positive))
    a.append(' '.join(negative))
    tfidf_matrix_twitter = tfidf_vectorizer.fit_transform(a)
    document_distance = (tfidf_matrix_twitter * tfidf_matrix_twitter.T)
    if document_distance.toarray()[0][1] > document_distance.toarray()[0][2]:
        label2.append('1')
    elif document_distance.toarray()[0][1] < document_distance.toarray()[0][2]:
        label2.append('-1')
    else:
        label2.append('0')

#%%
comment_valid=pd.DataFrame(columns=['comment','단어포함여부','유사도측정'])
comment_valid['comment']=words2
comment_valid['단어포함여부']=label
comment_valid['유사도측정']=label2

comment_valid['단어포함여부'].value_counts().plot(kind='bar')

#%%
# 실제 댓글과 비교
not_label = comment_valid['유사도측정'] != '0' # 라벨이 '0'인 경우(중립) 제외
comment_valid2 = comment_valid[not_label]














































