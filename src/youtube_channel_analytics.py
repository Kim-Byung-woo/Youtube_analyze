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
#%%
# 조회수별 구독자 변화 시각화
# load file
xlxs_dir = '오킹TV_channel_info.xlsx'
df_read_channel_info = pd.read_excel(xlxs_dir)
df_read_channel_info = df_read_channel_info.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제

plt.rc('font', size = 20, family='gulim') 
fig, ax0 = plt.subplots(figsize=(25, 10))
ax1 = ax0.twinx()
ax0.set_title("조회수별 구독자 변화")
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
# 영상개수별 조회수 시각화
xlxs_dir = '이스타TV_upload_cnt.xlsx'
df_read_channel_info = pd.read_excel(xlxs_dir)
df_read_channel_info = df_read_channel_info.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제

df_read_upload_view = df_read_channel_info.groupby("upload_date").sum() # 날짜별로 영상 조회수 합산
df_read_upload_cnt = pd.read_excel(xlxs_dir, sheet_name = 'upload count') # 날짜별 영상 개수 불러오기

df_upload_info = pd.merge(df_read_upload_view, df_read_upload_cnt, on = 'upload_date') # 날짜별로 영상 조회수, 개수 분류
df_anal = df_upload_info.groupby("daily video count").mean() # 영상 개수별 평균값 추출
df_anal = df_anal.reset_index()

sns.barplot(
    data= df_anal,
    x = "daily video count",
    y= "view count")
plt.show()

#%%
# 영상길이별 조회수 시각화
xlxs_dir = '오킹tv_upload_cnt.xlsx'
df_read_channel_info = pd.read_excel(xlxs_dir)
df_read_channel_info = df_read_channel_info.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제

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



df_anal = df_read_channel_info.groupby("rt_label")['view count'].mean() # 영살길이별 영상 조회수 평균
df_anal = df_anal.reset_index()
df_anal['rt_label count'] = list(df_read_channel_info.groupby("rt_label")['rt_label'].count()) # 영상 라벨링 별로 개수
df_anal = df_anal.drop(df_anal[df_anal['rt_label count'] < 2].index) # 영상 개수가 10개미만 라벨 행 삭제


sns.barplot(
    data= df_anal,
    x = "rt_label",
    y= "view count")
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

