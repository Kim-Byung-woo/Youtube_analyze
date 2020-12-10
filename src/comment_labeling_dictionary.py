# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:54:28 2020

@author: User
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import re
import time
import codecs
import os
#%%
## 파일에서 단어를 불러와 posneg리스트를 만드는 코드

positive = [] 
negative = [] 
posneg = []

file_dir = os.getcwd() # 현재 파일 경로 추출
file_dir = os.path.dirname(file_dir) # 상위 경로 추출 - 코드 파일과 단어 리스트 파일 위치가 틀려서

# pos = codecs.open("./positive_words_self.txt", 'rb', encoding='UTF-8') 
pos = codecs.open(file_dir + "/data/pos_pol_word.txt", 'rb', encoding='UTF-8') 

while True: 
    line = pos.readline()
    
    if not line: break 
    
    line = line.replace('\n', '') 
    positive.append(line) 
    posneg.append(line) 

pos.close()
 
neg = codecs.open(file_dir + "/data/neg_pol_word.txt", 'rb', encoding='UTF-8')

while True: 
    line = neg.readline() 
    
    if not line: break 
    
    line = line.replace('\n', '') 
    negative.append(line) 
    posneg.append(line) 
      
neg.close()
#%%
# 직접 작성한 단어장 불러오기
xlxs_dir = file_dir + '/data/my_word_book.xlsx'
df_my_word_book = pd.read_excel(xlxs_dir)
#%%
# 긍정단어장에서 긍정 형태소 추출
from konlpy.tag import *    
okt = Okt()  

list_positive = []
for i in positive:
    b = okt.morphs(i, norm = True, stem = True) # morphs: 형태소 추출
    
    for j in range(len(b)):
        list_positive.append(b[j]) # 긍정 단어장의 추출된 형태소를 list에 추가

# 중복 제거
set_X = set(list_positive)
list_positive = list(set_X)

# 불용어 불러오기
stopwords = []
stopwords = list(df_my_word_book['stopwords'])
# 불용어 중복 제거 (불용어 단어 중 중복이 있을수 있으므로)
set_words = set(stopwords)
stopwords = list(set_words)
stopwords = [x for x in stopwords if str(x) != 'nan']
# 불용어 제거
temp_X = [] 
temp_X = [word for word in list_positive if not word in stopwords]
list_positive = temp_X # 원본 데이터에 적용

# 분석에 어긋나는 특수문자, 의성어 제외
han = re.compile('[^ 가-힣]+') # 한글(자음과 모음이 따로 있는 경우도 포함)과 띄어쓰기를 제외한 모든 글자
temp_X = [] # 전처리된 댓글 리스트
for i in list_positive:
    tokens = re.sub(han,"",str(i)) # 한글(자음과 모음이 따로 있는 경우도 포함)과 띄어쓰기를 제외한 모든 글자
    temp_X.append(tokens)
list_positive = temp_X # 원본 데이터에 적용


# 공백만 있는 경우 제거
temp_X = [] # 전처리된 댓글 리스트
temp_X = [d.strip() for d in list_positive]
list_positive = [x for x in temp_X if str(x) != '']


# 형태소의 길이가 1 이하인 경우 제거
temp_X = [] # 형태소 리스트 초기화
for idx in range(len(list_positive)):
    length = len(list_positive[idx])
    if not length <= 1:
        temp_X.append(list_positive[idx])
list_positive = temp_X # 원본 데이터에 적용


# 기존 단어장에서 삭제할 긍정 단어 불러오기
delwords = []
delwords = list(df_my_word_book['del_poswords'])
# 삭제할 긍정 단어 중복 제거 (삭제할 긍정 단어 중 중복이 있을수 있으므로)
set_words = set(delwords)
delwords = list(set_words)
delwords = [x for x in delwords if str(x) != 'nan']
# 댓글에서 긍정적인 의미가 아닌 단어 삭제
temp_X = [] 
temp_X = [word for word in list_positive if not word in delwords]
list_positive = temp_X # 원본 데이터에 적용

# 기존 단어장에서  추가할 긍정 단어 불러오기
joinwords = []
joinwords = list(df_my_word_book['add_poswords'])
# 추가할 긍정 단어 중복 제거 (추가할 긍정 단어 중 중복이 있을수 있으므로)
set_words = set(joinwords)
joinwords = list(set_words)
joinwords = [x for x in joinwords if str(x) != 'nan']
# 긍정적인 의미 단어 추가
list_positive = list_positive + joinwords
# 중복 제거 (단어 추가 중 중복 될 수 있으므로)
set_X = set(list_positive)
list_positive = list(set_X)
#%%
# 부정단어장에서 부정 형태소 추출
from konlpy.tag import *    
okt = Okt()  

list_negative = []
for i in negative:
    b = okt.morphs(i, norm = True, stem = True) # morphs: 형태소 추출
    
    for j in range(len(b)):
        list_negative.append(b[j]) # 부정 단어장의 추출된 형태소를 list에 추가

# 중복 제거
set_X = set(list_negative)
list_negative = list(set_X)

# 불용어 불러오기
stopwords = []
stopwords = list(df_my_word_book['stopwords'])
# 불용어 중복 제거 (불용어 단어 중 중복이 있을수 있으므로)
set_words = set(stopwords)
stopwords = list(set_words)
stopwords = [x for x in stopwords if str(x) != 'nan']
# 불용어 제거
temp_X = [] 
temp_X = [word for word in list_negative if not word in stopwords]
list_negative = temp_X # 원본 데이터에 적용

# 분석에 어긋나는 특수문자, 의성어 제외
han = re.compile('[^ 가-힣]+') # 한글(자음과 모음이 따로 있는 경우도 포함)과 띄어쓰기를 제외한 모든 글자
temp_X = [] # 전처리된 댓글 리스트
for i in list_negative:
    tokens = re.sub(han,"",str(i)) # 한글(자음과 모음이 따로 있는 경우도 포함)과 띄어쓰기를 제외한 모든 글자 제거
    temp_X.append(tokens)
list_negative = temp_X # 원본 데이터에 적용

# 공백만 있는 경우 제거
temp_X = [] # 전처리된 댓글 리스트
temp_X = [d.strip() for d in list_negative]
list_negative = [x for x in temp_X if str(x) != '']


# 형태소의 길이가 1 이하인 경우 제거
temp_X = [] # 형태소 리스트 초기화
for idx in range(len(list_negative)):
    length = len(list_negative[idx])
    if not length <= 1:
        temp_X.append(list_negative[idx])
list_negative = temp_X # 원본 데이터에 적용

# 기존 단어장에서 삭제할 부정 단어 불러오기
delwords = []
delwords = list(df_my_word_book['del_negwords'])
# 삭제할 부정 단어 중복 제거 (삭제할 부정 단어 중 중복이 있을수 있으므로)
set_words = set(delwords)
delwords = list(set_words)
delwords = [x for x in delwords if str(x) != 'nan']
# 댓글에서 부정적인 의미가 아닌 단어 추가 삭제
temp_X = [] 
temp_X = [word for word in list_negative if not word in delwords]
list_negative = temp_X # 원본 데이터에 적용

# 기존 단어장에서 추가할 부정 단어 불러오기
joinwords = []
joinwords = list(df_my_word_book['add_negwords'])
# 추가할 부정 단어 중복 제거 (추가할 부정 단어 중 중복이 있을수 있으므로)
set_words = set(joinwords)
joinwords = list(set_words)
joinwords = [x for x in joinwords if str(x) != 'nan']
# 부정적인 의미 단어 추가
list_negative = list_negative + joinwords
# 중복 제거 (단어 추가 중 중복 될 수 있으므로)
set_X = set(list_negative)
list_negative = list(set_X)
#%%
# 크롤링한 댓글 불러오기
xlxs_dir = file_dir + '/data/comment_crwaling_sample_pos.xlsx'

df_video_info = pd.read_excel(xlxs_dir, sheet_name = 'video')
df_comment = pd.read_excel(xlxs_dir, sheet_name = 'comment')

print(len(df_comment))
#%%
# 댓글 전처리
# 댓글 중 중복값 제거
df_comment.drop_duplicates(subset=['comment'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
# 댓글 중 한글과 공백을 제외하고 모두 제거
df_comment['comment'] = df_comment['comment'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 댓글 중 공백만 있는 경우 제거
df_comment['comment'] = df_comment['comment'].str.strip() # strip: 문자열에서 양쪽 끝에 있는 공백과 \n 기호를 삭제
# 댓글 중 모두 제거된 데이터는 Nan값으로 대체
df_comment['comment'].replace('', np.nan, inplace=True)

# 댓글 중 Nan값 제거
df_comment = df_comment.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(df_comment.isnull().values.any()) # Null 값이 존재하는지 확인

df_comment.reset_index(inplace = True) # 행제거 인덱스도 같이 삭제되어 for문을 돌리기 위해서 인덱스 컬럼 초기화
df_comment = df_comment[['comment id', 'comment']] # 기존 인덱스 컬럼 삭제
print('전처리 후 댓글 개수: ', len(df_comment))

# 댓글 중 중복값 제거 - 한글외 다른 문자 제외 중 중복값 발생해서 중복제거를 2번 합니다.
df_comment.drop_duplicates(subset=['comment'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거

list_prep_comment = [] # 전처리된 댓글 리스트
list_prep_comment = df_comment['comment']
#%%
# 네이버 맞춤법 검사 후 수정 - 상당히 오래 걸림
#from hanspell import spell_checker
#list_prep_comment = [spell_checker.check(x).checked for x in list_prep_comment]
#%% 형태소 추출
from konlpy.tag import *

okt = Okt()  
list_okt = []
for i in list_prep_comment:
    b = okt.morphs(i, norm = True, stem = True) # morphs: 형태소 추출
    list_okt.append(b) # 추출된 형태소를 list에 추가
#%%
list_label = []
list_pos_text = []
list_neg_text = []


for idx in range(len(list_okt)):
    pos_score = 0
    list_text = []
    for text in list_positive: # 긍정 단어 목록을 1개씩 불러옵니다.
        cnt = list_okt[idx].count(text) # 긍정 단어가 형태소 목록에 몇개 있는지 파악
        pos_score += cnt 
        
        if cnt != 0: # 추출된 형태소 목록에 긍정 단어를 찾았을 경우
            list_text.append(text) # 찾은 긍정 단어 누적            
    list_pos_text.append(list_text) # 찾은 긍정 단어 목록 저장
    
    neg_score = 0
    list_text = []
    for text in list_negative: # 부정 단어 목록을 1개씩 불러옵니다.
        cnt = list_okt[idx].count(text) # 긍정 단어가 형태소 목록에 몇개 있는지 파악
        neg_score += cnt 
        
        if cnt != 0: # 추출된 형태소 목록에 긍정 단어를 찾았을 경우
            list_text.append(text) # 찾은 긍정 단어 누적            
    list_neg_text.append(list_text) # 찾은 부정 단어 목록 저장

    # 0: negative 1: positve 2: none
    label = -1
    if pos_score > neg_score:
        label = 1
    elif pos_score < neg_score:
        label = 0
    else:
        label = 2
    
    list_label.append(label)

df_comment['morphs'] = list_okt # 형태소 추출 결과
#df_comment['pos score'] = list_pos_socre # 찾은 긍정 단어 개수
df_comment['pos text'] = list_pos_text # 찾은 긍정 단어 목록
#df_comment['neg score'] = list_neg_score # 찾은 부정 단어 개수    
df_comment['neg text'] = list_neg_text # 찾은 부정 단어 목록
df_comment['okt label'] = list_label # 라벨링 결과
df_okt = df_comment.groupby(by = ['okt label'], as_index = False).count()
#%%
filename = '부정 단어장 수정.xlsx'
df_comment.to_excel(filename)
#%%
df_none = df_comment[df_comment['okt label'] == 1] # 중립인 댓글 추출
filename = '긍정_긍정 단어장 수정.xlsx'
df_none.to_excel(filename)




































