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
import os
#%%
file_dir = os.getcwd() # 현재 파일 경로 추출
file_dir = os.path.dirname(file_dir) # 상위 경로 추출 - 코드 파일과 단어 리스트 파일 위치가 틀려서

# 일일 구독자수에 따른 조회수 시각화
# load file
xlxs_dir = file_dir + '/data/이스타TV_channel_info.xlsx'

df_read_channel_info = pd.read_excel(xlxs_dir)
#df_read_channel_info = df_read_channel_info.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제

plt.rc('font', size = 20, family='gulim')
fig, ax0 = plt.subplots(figsize=(25, 10))
ax1 = ax0.twinx()
ax0.set_title("일일 구독자수에 따른 조회수 변화")
ax0.plot(df_read_channel_info['daily subscribe count'], 'r-', label= '일일 구독자 변화')
ax0.set_ylabel("일일 구독자 변화")
ax0.grid(False)
ax0.legend(loc = 1) # 범주
ax1.plot(df_read_channel_info['daily view count'], 'g-', label='일일 조회수')
ax1.set_ylabel("일일 조회수")
ax1.grid(False)
ax1.legend(loc = 2) # 범주
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
ax0.set_ylabel("누적 구독자 변화")
ax0.grid(False)
ax0.legend(loc = 1) # 범주
ax1.plot(df_read_channel_info['daily view count'], 'g-', label='일일 조회수')
ax1.set_ylabel("일일 조회수")
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

