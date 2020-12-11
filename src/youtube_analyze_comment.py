# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:45:47 2020

@author: user

"""
#%%
import re
 
text = u'This dog \U0001f602'
print(text) # with emoji
 
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
print(emoji_pattern.sub(r'', text)) # no emoji


'''
특정 이모지만 제거하는 것이 아니라 BMP영역 이외 문자를 제거하는 것으로 바꿔야 할 것 같다.
0000-FFFF 까지가 BMP이고 10FFFF까지 SMP, SIP, TIP, SSP, PUA 공간이 잡혀있어서
10000-10FFFF까지 제거하는 것으로 코드를 바꿔야한다.
'''
text = '안녕하세요 반갑습니다🐶'
print(text) 
only_BMP_pattern = re.compile("["
        u"\U00010000-\U0010FFFF"  #BMP characters 이외
                           "]+", flags=re.UNICODE)
print(only_BMP_pattern.sub(r'', text))# BMP characters만
#%%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import re
import time
from collections import Counter
from wordcloud import WordCloud
from konlpy.tag import *
import os

def get_noun(comment_txt):
    okt = Okt() 
    noun = []
    
    if len(comment_txt)>0:
        tw = okt.pos(comment_txt)
        for i,j in tw:
            if j == 'Noun':
                noun.append(i)
    return noun
#%%
# load file
# 데이터셋 로드
file_dir = os.getcwd() # 현재 파일 경로 추출
file_dir = os.path.dirname(file_dir) # 상위 경로 추출

xlxs_dir = file_dir + '/data/엘론_video_info.xlsx'

df_read_video_info = pd.read_excel(xlxs_dir, sheet_name = 'video')
df_read_comment = pd.read_excel(xlxs_dir, sheet_name = 'comment')

for idx in range(len(df_read_comment)):
    comment = df_read_comment.iloc[idx]['comment']
    print(comment)
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
    tokens = re.sub(emoji_pattern,"",i)
    tokens = re.sub(han,"",tokens)
    comment_result.append(tokens)

df_comment_result = pd.DataFrame(comment_result, columns=["comment"])

# 명사 추출
df_comment_result['token'] = df_comment_result['comment'].apply(lambda x: get_noun(x))

noun_list = []
for i in range(len(df_comment_result)):
    for j in range(len(df_comment_result['token'].iloc[i])):
        noun_list.append(df_comment_result['token'].iloc[i][j])
        
counts = Counter(noun_list) # 추출된 명사 빈도수 확인
tags = counts.most_common(30) # 빈도수 상위 30개 추출
#%%
#bar chart
test = pd.DataFrame({'word':[],
                    'count':[]})
for i in range(len(tags)):
    word = tags[i][0]
    count = tags[i][1]
    
    insert_data = pd.DataFrame({'word':[word],
                                'count':[count]})
    test = test.append(insert_data)

test.index = range(len(test))

index = np.arange(len(test))
plt.rc('font', family='Malgun Gothic') # 한글 깨짐 수정
plt.figure(figsize=(60, 32))
plt.bar(index,test['count'].tolist() )
plt.xlabel('word', fontsize=5)
plt.ylabel('count', fontsize=5)
plt.xticks(index, test['word'].tolist(), fontsize=30, rotation=30)
plt.title('단어 빈도수 시각화')
plt.show()
#%%
#wordcloud
wc = WordCloud(font_path = file_dir + '/font/NanumBarunGothic.ttf',background_color='white', width=800, height=600)
cloud = wc.generate_from_frequencies(dict(tags))
plt.figure(figsize=(10, 8))
plt.axis('off')
plt.imshow(cloud)
plt.show()
#%%
# 댓글 감성분석 시각화
# 데이터셋 로드
xlxs_dir = file_dir + '/data/감성분석 모델 비교 결과.xlsx'
df_sentiment_result = pd.read_excel(xlxs_dir)

comment_cnt = len(df_sentiment_result) - 1 # 총 댓글 개수, -1은 컬령명 제외

# 정확성이 70% 보다 크고 라벨링 0(부정)인 댓글 추출
df_temp = df_sentiment_result[(df_sentiment_result['accuracy_review'] > 70) & (df_sentiment_result['label_review'] == 0)]
neg_cnt_by_review = len(df_temp) # 영화 리뷰 모델 기반 긍정 댓글 개수

# 정확성이 70% 보다 크고 라벨링 0(부정)인 댓글 추출
df_temp = df_sentiment_result[(df_sentiment_result['accuracy_review'] > 70) & (df_sentiment_result['label_review'] == 1)]
pos_cnt_by_review = len(df_temp) # 영화 리뷰 모델 기반 부정 댓글 개수

# 정확성이 70% 미만인 댓글 추출
df_temp = df_sentiment_result[(df_sentiment_result['accuracy_review'] < 70)]
non_cnt_by_review = len(df_temp) # 영화 리뷰 모델 기반 중립 댓글 개수

# pie chart
## 데이터 준비
labels = ['Positive Comment','Negative Comment','Neutrality Comment'] ## 라벨
frequency = [pos_cnt_by_review, neg_cnt_by_review, non_cnt_by_review] ## 빈도
colors = ['#87ceeb', '#ff9999', '#d395d0']

fig = plt.figure(figsize=(8,8)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 배경색을 하얀색으로 설정
ax = fig.add_subplot() ## 프레임 생성
 
pie = ax.pie(frequency, ## 파이차트 출력
       startangle = 90, ## 시작점을 90도(degree)로 지정
       explode = (0.05 , 0, 0), # 간격 설정
       #shadow=True, # 그림자 설정
       counterclock = False, ## 시계 방향으로 그린다.
       autopct = lambda p : '{:.2f}%'.format(p), ## 퍼센티지 출력
       colors = colors, ## 색상 지정
       wedgeprops = {'edgecolor':'k','linewidth': 1} ## 테두리 속성 지정
       )
 
plt.legend(pie[0],labels) ## 범례 표시
plt.show()

# 정확성이 70% 보다 크고 라벨링 0(부정)인 댓글 추출
df_temp = df_sentiment_result[(df_sentiment_result['accuracy_comment'] > 70) & (df_sentiment_result['label_comment'] == 0)]
neg_cnt_by_comment = len(df_temp) # 영화 리뷰 모델 기반 긍정 댓글 개수

# 정확성이 70% 보다 크고 라벨링 0(부정)인 댓글 추출
df_temp = df_sentiment_result[(df_sentiment_result['accuracy_comment'] > 70) & (df_sentiment_result['label_comment'] == 1)]
pos_cnt_by_comment = len(df_temp) # 영화 리뷰 모델 기반 부정 댓글 개수

# 정확성이 70% 미만인 댓글 추출
df_temp = df_sentiment_result[(df_sentiment_result['accuracy_comment'] < 70)]
non_cnt_by_comment = len(df_temp) # 영화 리뷰 모델 기반 중립 댓글 개수

# pie chart
## 데이터 준비
frequency = [pos_cnt_by_comment, neg_cnt_by_comment, non_cnt_by_comment] ## 빈도

fig = plt.figure(figsize=(8,8)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 배경색을 하얀색으로 설정
ax = fig.add_subplot() ## 프레임 생성
 
pie = ax.pie(frequency, ## 파이차트 출력
       startangle = 90, ## 시작점을 90도(degree)로 지정
       explode = (0.05 , 0, 0), # 간격 설정
       #shadow=True, # 그림자 설정
       counterclock = False, ## 시계 방향으로 그린다.
       autopct = lambda p : '{:.2f}%'.format(p), ## 퍼센티지 출력
       colors = colors, ## 색상 지정
       wedgeprops = {'edgecolor':'k','linewidth': 1} ## 테두리 속성 지정
       )
 
plt.legend(pie[0],labels) ## 범례 표시
plt.show()

























