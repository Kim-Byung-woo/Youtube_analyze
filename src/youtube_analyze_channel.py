# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 00:25:23 2020

@author: user
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import os
from konlpy.tag import *

file_dir = os.getcwd() # 현재 파일 경로 추출
file_dir = os.path.dirname(file_dir) # 상위 경로 추출 - 코드 파일과 단어 리스트 파일 위치가 틀려서
#%%



plt.rc('font', size = 20, family='gulim')
plt.plot(df_read_channel_info['daily subscribe count'], 'r-', label= '일일 구독자 변화')
plt.plot(df_read_channel_info['daily view count'], 'g-', label='일일 조회수')

plt.xticks([0, 1, 2])

plt.show()



#%%

# 일일 구독자수에 따른 조회수 시각화
# load file
xlxs_dir = file_dir + '/data/이스타TV_channel_info.xlsx'

df_read_channel_info = pd.read_excel(xlxs_dir)
#df_read_channel_info = df_read_channel_info.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제

plt.rc('font', size = 20, family='gulim')
fig, ax0 = plt.subplots(figsize=(25, 10))

ax0.set_title("일일 구독자수 증가에 따른 조회수 변화")
ax0.plot(df_read_channel_info['daily subscribe count'], 'r-', label= '일일 구독자 변화')

#ax0.xticks(['19년 8월', '19년 9월', '19년 10월', '19년 11월', '19년 12월', '20년 1월', '20년 2월', '20년 3월', '20년 4월', '20년 5월', '20년 6월', '20년 7월'])
ax0.set_ylabel("일일 구독자 증가")
ax0.grid(False)
ax0.legend(loc = 1) # 범주


ax1 = ax0.twinx()
ax1.plot(df_read_channel_info['daily view count'], 'g-', label='일일 조회수')
ax1.set_ylabel("일일 조회수")
ax1.grid(False)
ax1.legend(loc = 2) # 범주

# x축 눈금 재설정
plt.xticks(np.arange(15, len(df_read_channel_info), 30), labels=['19년 8월', '19년 9월', '19년 10월', '19년 11월', '19년 12월', '20년 1월', '20년 2월', '20년 3월', '20년 4월', '20년 5월', '20년 6월', '20년 7월'])
plt.show()
#%%
# 누적 구독자수에 따른 조회수 시각화
xlxs_dir = file_dir + '/data/이스타TV_channel_info.xlsx'

df_read_channel_info = pd.read_excel(xlxs_dir)
df_read_channel_info = df_read_channel_info.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제

plt.rc('font', size = 20, family='gulim')
fig, ax0 = plt.subplots(figsize=(25, 10))
ax1 = ax0.twinx()
ax0.set_title("누적 구독자수에 따른 조회수 변화")
ax0.plot(df_read_channel_info['subscribe count'], 'r-', label= '누적 구독자 변화')
ax0.set_ylabel("누적 구독자 변화량")
ax0.grid(False)
ax0.legend(loc = 1) # 범주
ax1.plot(df_read_channel_info['daily view count'], 'g-', label='일일 조회수')
ax1.set_ylabel("일일 조회수 증가량")
ax1.grid(False)
ax1.legend(loc = 2) # 범주
plt.show()
#%%
# 일일 영상 업로드 개수별 조회수 시각화
# load file
xlxs_dir = file_dir + '/data/이스타TV_upload_cnt.xlsx'

df_read_channel_info = pd.read_excel(xlxs_dir)
# df_read_channel_info = df_read_channel_info.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제

df_read_upload_view = df_read_channel_info.groupby("upload date").sum() # 날짜별로 영상 조회수 합산
df_read_upload_cnt = pd.read_excel(xlxs_dir, sheet_name = 'upload count') # 날짜별 영상 개수 불러오기

df_upload_info = pd.merge(df_read_upload_view, df_read_upload_cnt, on = 'upload date') # 날짜별로 영상 조회수, 개수 분류
# 영상 개수별 평균값 추출
df_anal = df_upload_info.groupby("daily video count")['view count'].sum() # 일일 영상 업로드 개수별 조회수 합산
df_anal = df_anal.reset_index() # seires에서 dataframe으로 타입 변경
df_anal['video count'] = list(df_upload_info.groupby("daily video count")['upload date'].count()) # 업로드 영상개수 별로 업로드 날짜 카운팅, list로 바인딩 안하면 1행씩 밀림

# 평균 조회수 = 총 조회수 / (일일 업로드 영상 개수 * 업로드 영상 개수별 날짜 개수) 
# ex. 평균 조회수 = 총 조회수 / (1 * 업로드 영상 개수가 1개인 날짜들의 개수)
df_anal['average view count'] = df_anal['view count'] / (df_anal['daily video count'] * df_anal['video count']) 

# 시각화
plt.rc('font', size = 10, family='gulim')
sns.barplot(
    data= df_anal,
    x = "daily video count",
    y = "average view count")
plt.xlabel('일일 영상 업로드 개수')
plt.ylabel('영상 평균 조회수')
plt.show()
#%%
# 영상 길이별 조회수 시각화
# load file
xlxs_dir = file_dir + '/data/이스타TV_upload_cnt.xlsx'

df_read_channel_info = pd.read_excel(xlxs_dir)
# df_read_channel_info = df_read_channel_info.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제

# 영살 길이별 라벨링
list_rt = df_read_channel_info['running time']
list_rt_label = []
for idx in range(len(list_rt)):
    str_rt = list_rt[idx].replace(':', '') # 15:27(15분 28초) -> '1527'
    int_rt = int(str_rt) # '1527' -> 1527
    rt_label = int(int_rt / 1000)
    list_rt_label.append(rt_label) # 1527 -> 1, 0846 -> 0
# 기존 데이터에 라벨링 추가
df_read_channel_info['rt_label'] = list_rt_label

df_anal = df_read_channel_info.groupby("rt_label")['view count'].sum() # 영살 길이 라벨별 조회수 합산 영살길이별 영상 조회수 평균
df_anal = df_anal.reset_index() # seires에서 dataframe으로 타입 변경
df_anal['video count'] = list(df_read_channel_info.groupby("rt_label")['rt_label'].count()) # 영살 길이별 영상 개수
df_anal = df_anal.drop(df_anal[df_anal['video count'] < 10].index) # 영상 개수가 10개미만 행 삭제
df_anal['average view count'] = df_anal['view count'] / (df_anal['video count']) # 평균 조회수 = 영상 길이별 조회수 합산 /  영상 길이별 영상 개수

# 시각화
plt.rc('font', size = 10, family='gulim')
sns.barplot(
    data= df_anal,
    x = "rt_label",
    y = "average view count")
plt.xlabel('영상 길이')
plt.ylabel('영상 평균 조회수')
plt.show()
#%%
'''
영상 제목, 영상 조회수 -> 영상 제목키워드 별 조회수 분석

# 1안
1. 모든 영상 제목 토큰화
2. 빈도수 상위 30개 단어 추출
3. 추출된 단어의 누적, 평균 조회수

# 2안
1. 상위 조회수(상위 25%) 영상 제목 토큰화
2. 빈도수 상위 30개 단어 추출
3. 각 단어별 누적, 평균 조회수 
'''

# load file
xlxs_dir = file_dir + '/data/이스타TV_upload_cnt.xlsx'
df_read_channel_info = pd.read_excel(xlxs_dir)



# 제목 전처리
# 제목 중 중복값 제거
df_read_channel_info.drop_duplicates(subset=['title'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
# 제목 중 한글(자음과 모음은 제거),영어, 숫자, 공백을 제외하고 모두 제거
df_read_channel_info['title'] = df_read_channel_info['title'].str.replace("[^a-zA-Z0-9가-힣 ]","")
# 제목 중 공백만 있는 경우 제거
df_read_channel_info['title'] = df_read_channel_info['title'].str.strip() # strip: 문자열에서 양쪽 끝에 있는 공백과 \n 기호를 삭제
# 제목 중 모두 제거된 데이터는 Nan값으로 대체
df_read_channel_info['title'].replace('', np.nan, inplace=True)
# 제목 중 Nan값 제거
df_read_channel_info = df_read_channel_info.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(df_read_channel_info.isnull().values.any()) # Null 값이 존재하는지 확인

# 1안
list_prep_comment = [] 
list_prep_comment = df_read_channel_info['title'] # 전처리된 제목 리스트 저장

okt = Okt()  

# 전처리된 제목에서 형태소 추출
list_okt = []
for i in list_prep_comment:
    b = okt.morphs(i, norm = True) # morphs: 형태소 추출
    list_okt.append(b) # 추출된 형태소를 list에 추가

# 추출된 형태소를 하나의 list에 저장
list_morphs = []
for i in list_okt:
   for j in range(len(i)):
       list_morphs.append(i[j]) 
      

# 불용어 리스트를 불러오기
stopwords = pd.read_excel(file_dir + '/data/title_stopwords.xlsx')['stopwords']
# 불용어 중복 제거 (불용어 단어 중 중복이 있을수 있으므로)
set_words = set(stopwords)
stopwords = list(set_words)
stopwords = [x for x in stopwords if str(x) != 'nan']
# 불용어 제거
temp_X = [] 
temp_X = [word for word in list_morphs if not word in str(stopwords)]
list_morphs = temp_X # 원본 데이터에 적용

counts = Counter(list_morphs) # 추출된 명사 빈도수 확인
most_morphs = counts.most_common(30) # 빈도수 상위 30개 추출
df_anal = pd.DataFrame(list(most_morphs), columns=['morphs', 'counts']) # 튜플 타입인 most_morphs를 Dataframe으로 형변환

# 단어별 평균 조회수 구하기
list_avg_view = []
for idx in range(len(df_anal)):
    view_cnt = df_read_channel_info[df_read_channel_info['title'].str.contains(df_anal['morphs'][idx])]['view count'].sum() # 추출된 상위 단어가 포함된 영상의 조회수 합산
    video_cnt = len(df_read_channel_info[df_read_channel_info['title'].str.contains(df_anal['morphs'][idx])]['view count']) # 추출된 상위 단어가 포함된 영상의 개수
    list_avg_view.append(view_cnt / video_cnt) # 평균 조회수 = (합산된 조회수 / 영상의 개수)

df_anal['average view count'] = list_avg_view # 평균 조회수 데이터 추가
df_anal = df_anal.sort_values(by=['average view count'], axis=0, ascending=False) # 평균 조회수 기준 내림차순으로 정렬

# 시각화
plt.figure(figsize=[20, 5])
plt.rc('font', size = 10, family='gulim')
sns.barplot(
    data= df_anal,
    x = "morphs",
    y = "average view count")
plt.xlabel('단어')
plt.ylabel('영상 평균 조회수')
plt.show()
#%%
# 2안
top_cnt = round(len(df_read_channel_info) * 0.3, 0) # 전체 영상의 30% 개수
df_temp = df_read_channel_info.sort_values(by=['view count'], axis=0, ascending=False) # 영상 조회수 기준 내림차순 정렬
df_temp = df_temp.head(int(top_cnt))


# 제목 전처리
# 제목 중 중복값 제거
df_temp.drop_duplicates(subset=['title'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
# 제목 중 한글(자음과 모음은 제거),영어, 숫자, 공백을 제외하고 모두 제거
df_temp['title'] = df_temp['title'].str.replace("[^a-zA-Z0-9가-힣 ]","")
# 제목 중 공백만 있는 경우 제거
df_temp['title'] = df_temp['title'].str.strip() # strip: 문자열에서 양쪽 끝에 있는 공백과 \n 기호를 삭제
# 제목 중 모두 제거된 데이터는 Nan값으로 대체
df_temp['title'].replace('', np.nan, inplace=True)
# 제목 중 Nan값 제거
df_temp = df_temp.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(df_temp.isnull().values.any()) # Null 값이 존재하는지 확인

list_prep_comment = [] 
list_prep_comment = df_temp['title'] # 전처리된 제목 리스트 저장

okt = Okt()  

# 전처리된 제목에서 형태소 추출
list_okt = []
for i in list_prep_comment:
    b = okt.morphs(i, norm = True) # morphs: 형태소 추출
    list_okt.append(b) # 추출된 형태소를 list에 추가

# 추출된 형태소를 하나의 list에 저장
list_morphs = []
for i in list_okt:
   for j in range(len(i)):
       list_morphs.append(i[j]) 
      

# 불용어 리스트를 불러오기
stopwords = pd.read_excel(file_dir + '/data/title_stopwords.xlsx')['stopwords']
# 불용어 중복 제거 (불용어 단어 중 중복이 있을수 있으므로)
set_words = set(stopwords)
stopwords = list(set_words)
stopwords = [x for x in stopwords if str(x) != 'nan']
# 불용어 제거
temp_X = [] 
temp_X = [word for word in list_morphs if not word in str(stopwords)]
list_morphs = temp_X # 원본 데이터에 적용

counts = Counter(list_morphs) # 추출된 명사 빈도수 확인
most_morphs = counts.most_common(30) # 빈도수 상위 30개 추출
df_anal = pd.DataFrame(list(most_morphs), columns=['morphs', 'counts']) # 튜플 타입인 most_morphs를 Dataframe으로 형변환

# 단어별 평균 조회수 구하기
list_avg_view = []
for idx in range(len(df_anal)):
    view_cnt = df_temp[df_temp['title'].str.contains(df_anal['morphs'][idx])]['view count'].sum() # 추출된 상위 단어가 포함된 영상의 조회수 합산
    video_cnt = len(df_temp[df_temp['title'].str.contains(df_anal['morphs'][idx])]['view count']) # 추출된 상위 단어가 포함된 영상의 개수
    list_avg_view.append(view_cnt / video_cnt) # 평균 조회수 = (합산된 조회수 / 영상의 개수)

df_anal['average view count'] = list_avg_view # 평균 조회수 데이터 추가
df_anal = df_anal.sort_values(by=['average view count'], axis=0, ascending=False) # 평균 조회수 기준 내림차순으로 정렬

# 시각화
plt.figure(figsize=[20, 5])
plt.rc('font', size = 10, family='gulim')
sns.barplot(
    data= df_anal,
    x = "morphs",
    y = "average view count")
plt.xlabel('단어')
plt.ylabel('영상 평균 조회수')
plt.show()
#%%
'''
=============== 날짜 표시 간소화 ===============
참조: https://www.python2.net/questions-363212.htm
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from datetime import datetime
from datetime import timedelta

months = MonthLocator()  # every month
fig, ax = plt.subplots(figsize=(100, 10))

### create sample data
your_df = pd.DataFrame()
your_df['vals'] = np.arange(1000)
## make sure your datetime is considered as such by pandas
your_df['date'] = pd.to_datetime([dt.today()+timedelta(days=x) for x in range(1000)])

your_df=  your_df.set_index('date') ## set it as index
### plot it
fig = plt.figure(figsize=[20, 5])
ax = fig.add_subplot(111)
ax.plot(your_df['vals'])
monFmt = DateFormatter('%Y-%m')   
plt.xticks(rotation='vertical')
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_major_formatter(monFmt)

li = []
val = datetime.strptime(df_read_channel_info.iloc[0]['date'], '%Y-%m-%d')
li = li.append(val)

