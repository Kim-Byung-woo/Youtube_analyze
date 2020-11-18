# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 13:59:33 2020

@author: user
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import re
import time
import codecs

#%%
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic') # Window 의 한글 폰트 설정
import seaborn as sns

x_data = np.array(['영희가 사랑하는 강아지 백구를 산책시키고 있다.',
        '철수가 사랑하는 소 누렁이를 운동시키고 있다.',
        '영희와 철수는 소와 강아지를 산책 및 운동시키고 있다.'])

twitter = Twitter()
for i, document in enumerate(x_data):
    nouns = twitter.nouns(document)    
    x_data[i] = ' '.join(nouns)
print(x_data) #['영희 사랑 강아지 백구 산책', '철수 사랑 소 누렁이 운동', '영희 철수 소 강아지 산책 및 운동']

vect = TfidfVectorizer()

x_data = vect.fit_transform(x_data)

cosine_similarity_matrix = (x_data * x_data.T)
print(cosine_similarity_matrix.shape) #(3, 3)
print(cosine_similarity_matrix)
'''
  (0, 1)    0.19212485958220318
  (0, 2)    0.5605318467638107
  (0, 0)    0.9999999999999999
  (1, 2)    0.4113054999991637
  (1, 1)    1.0000000000000002
  (1, 0)    0.19212485958220318
  (2, 1)    0.4113054999991637
  (2, 2)    0.9999999999999999
  (2, 0)    0.5605318467638107
'''
print(cosine_similarity_matrix.toarray())
'''
[[1.         0.19212486 0.56053185]
 [0.19212486 1.         0.4113055 ]
 [0.56053185 0.4113055  1.        ]]
'''

#

sns.heatmap(cosine_similarity_matrix.toarray(), cmap='viridis')
#sns.heatmap(cosine_similarity_matrix.toarray(), xticklabels=x_data, yticklabels=x_data, cmap='viridis')
plt.show()


#%%
# 파이썬 표준 라이브러리인 difflib의 SequenceMatcher를 사용해서 두 개의 문자열의 유사성을 수치화할 수 있습니다.
from difflib import SequenceMatcher
from konlpy.tag import Okt
from collections import Counter

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

similar("잘해","못해")
similar("감사","감사하다")
similar("긍정적","부정적") # 길이로만 구분하여 이상한 결과가 나옴
similar("지양적","지향적") # 길이로만 구분하여 이상한 결과가 나옴

str_test = '너무 감사합니다'
str_base = '감사'

okt = Okt()
o_noun = okt.nouns(str_test)
o_morphs = okt.morphs(str_test)

print(okt.pos(str_test, norm = True, stem = True))

for idx in range(len(o_morphs)):
    int_simr = similar(o_morphs[idx], str_base)
    print(f'{o_morphs[idx]}와 {str_base}의 유사도 = {int_simr}')
#%%
import numpy as np

def cos_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
    similarity = dot_product / l2_norm     
    
    return similarity

from sklearn.feature_extraction.text import TfidfVectorizer

doc_list = ['if you take the blue pill, the story ends' ,
            'if you take the red pill, you stay in Wonderland',
            'if you take the red pill, I show you how deep the rabbit hole goes']

tfidf_vect_simple = TfidfVectorizer()
feature_vect_simple = tfidf_vect_simple.fit_transform(doc_list)

print(feature_vect_simple.shape)
print(type(feature_vect_simple))

# TFidfVectorizer로 transform()한 결과는 Sparse Matrix이므로 Dense Matrix로 변환. 
feature_vect_dense = feature_vect_simple.todense()

#첫번째 문장과 두번째 문장의 feature vector  추출
vect1 = np.array(feature_vect_dense[0]).reshape(-1,)
vect2 = np.array(feature_vect_dense[1]).reshape(-1,)

#첫번째 문장과 두번째 문장의 feature vector로 두개 문장의 Cosine 유사도 추출
similarity_simple = cos_similarity(vect1, vect2)
print('문장 1, 문장 2 Cosine 유사도: {0:.3f}'.format(similarity_simple))


#%%
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

doc_test = []
doc_test.append('영희 무료 철수 배신 실망') # 댓글에 추출된 형태소 혹은 명사
doc_test.append('최고 축하 대박 응원 무료 영희 무료 무료') # 긍정 단어 목록
doc_test.append('영호 실망 배신 사기 분노 스타 사과 치약') # 부정 단어 목록

import konlpy 
from konlpy.tag import Okt 
okt=Okt()
word=[]
for sentence in doc_test:
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 형태소룰 추출
    word.append(temp_X) # 추출된 형태소를 list에 추가
    # 토큰화 temp_X = [word for word in temp_X if not word in stopwords] 
    # 불용어 제거 X_train.append(temp_X) X_test = [] for sentence in test_data['title']: temp_X = [] temp_X = okt.morphs(sentence, stem=True) # 토큰화 temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 X_test.append(temp_X)
    
words=[]
for i in word:
    words.append(' '.join(i)) # 분리된 형태소를 문자열로 바인딩 후 리스트에 추가

# TF-IDF는 가중치를 구하는 알고리즘 -> 단어 발생 빈도인 TF에서, 전체 발생 횟수에 따른 패널티를 부여해준 개념이 바로 TF-IDF
# 예를 들어 1.문서간의 비슷한 정도, 2.문서 내 단어들에 척도를 계산하여 핵심어를 추출
# TF는 간단하게 말해서 문서 내 특정 단어의 빈도를 말합니다.
# DF는 해당 단어가 나타난 문서의 수
# IDF는 DF값에 역수
# 특정 단어 T가 모든 문서에 등장하는 흔한 단어라면 TF-IDF 가중치는 낮춰줍니다.

tfidf_vectorizer = TfidfVectorizer(min_df=1) # min_df: 최소값 min_df = 2인 경우 빈도가 1번인 단어는 제외

# Bag of Words란 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법입니다.
# 벡터화: 행렬을 N by 1 형태로 변환 -> 특징을 수치화 하여 계산을 빠르게 하기 위해서
# 문장 내 단어들을 tf-idf 값으로 가중치를 설정하여 BOW 벡터를 만든다.
tfidf_matrix_twitter = tfidf_vectorizer.fit_transform(words)

# TF-IDF행렬과 TF-IDF 전치행렬을 곱하면 유사도를 구할수 있습니다.
# 행렬의 곱에서 나온 유사도는 cosine_similarity 함수의 결과값과 같다.
document_distance = (tfidf_matrix_twitter * tfidf_matrix_twitter.T) 
print(document_distance.toarray())

from sklearn.metrics.pairwise import cosine_similarity
cosine_distance = cosine_similarity(tfidf_matrix_twitter[0:1], tfidf_matrix_twitter[1:2])
#print(cosine_distance)
cosine_distance = cosine_similarity(tfidf_matrix_twitter[0:1], tfidf_matrix_twitter[2:3])
#print(cosine_distance)
cosine_distance = cosine_similarity(tfidf_matrix_twitter[1:2], tfidf_matrix_twitter[2:3])
#print(cosine_distance)




#%%
