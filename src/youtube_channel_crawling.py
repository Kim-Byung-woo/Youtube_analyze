# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:54:44 2020

@author: A
"""
#%%
'''
============== Read me ================

video_url 변수에 크롤링 하고 싶은 유튜브 영상의 주소를 저장 후 프로그램을 실행해야 합니다.
1. https://www.youtube.com에 점속
2. 사용자가 원하는 채널을 클릭합니다.
3. 해당 채널id를 복사 - 채널 id -> 채널 주소에서 channel뒤에 있는 코드
3-1.예시: https://www.youtube.com/channel/UCOH52Yqq4-rdLvpt2Unsqsw에서 'UCOH52Yqq4-rdLvpt2Unsqsw'가 해당 채널의 id 입니다.
4. channel_url에 복사된 채널id를 저장.
5. 프로그램(크콜링) 실행 -자동으로 크롬 드라이버 실행되오니 드라이버를 강제 종료 하지 마시오
5.1 크롬 드라이버를 최신화 하면 진행되지 않습니다. 반드시 최소화는 하지 않습니다.
6. 저장된 xlsx 파일 확인

'''
#%%#
import numpy as np
import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# Chrome Driver 실행
path = 'chromedriver.exe'
driver = webdriver.Chrome(path)

noxinfluencer_url = 'https://kr.noxinfluencer.com/youtube/channel/'
# 크리에이터 채널 id
channel_url = 'UCom6YhUY62jM52nIMjf5_dw' # UCaVGMXXmaE2hO9qXu09GwdQ

delay = 3
driver.implicitly_wait(delay)
driver.get(noxinfluencer_url + channel_url)
#driver.maximize_window()
time.sleep(delay)

username = driver.find_element_by_xpath('/html/body/section[1]/div/div[2]/p/span[1]').text

# page down scroll
body = driver.find_element_by_tag_name('body')
num_of_pagedowns = 2
while num_of_pagedowns:
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(1.5)
    num_of_pagedowns -= 1

# 구독자 누적 차트에 마우스 호버 실행
element_to_hover_over = driver.find_element_by_xpath('//*[@id="channel-history-sub-chart"]/div[1]/canvas')
hover = ActionChains(driver).move_to_element(element_to_hover_over) # 차트의 가운데로 마우스 호버
hover.perform()

# 차트 사이즈 가져오기
element_size = element_to_hover_over.size
element_height = element_size['height']
element_width = element_size['width']

# 차트의 좌측으로 이동
x_offset = (element_width / 2)
move = ActionChains(driver).move_by_offset(-x_offset, 0)
move.perform()

# 차트의 우측으로 이동하면서 날짜/누적 구독자수 저장
'''
soup = BeautifulSoup(html, 'lxml') -> 사용해서 크롤링 시간 단축 해보기!
''' 
list_sub_date = []
list_subscribe_cnt = []
for idx in range(int(element_width / 2)):
    # 마우스 이동
    move = ActionChains(driver).move_by_offset(2, 0)
    move.perform()
    
    chart_info = driver.find_element_by_xpath('//*[@id="channel-history-sub-chart"]/div[2]').text
    if chart_info.find('\n') != -1 :
        list_sub_date.append(chart_info.split('\n')[0])
        
        subscribe_cnt = chart_info.split('\n')[1]
        if subscribe_cnt.find('만') != -1:
            subscribe_cnt = subscribe_cnt.replace('만', '')
            subscribe_cnt = float(subscribe_cnt) * 10000
        elif subscribe_cnt.find('천') != -1:
            subscribe_cnt = subscribe_cnt.replace('천', '')
            subscribe_cnt = subscribe_cnt * 1000
        else:
            subscribe_cnt = float(subscribe_cnt)
        list_subscribe_cnt.append(subscribe_cnt)
            
'''
# 일별 버튼 클릭
more_button = driver.find_element_by_xpath('//*[@id="tab-channel"]/div[4]/div[1]/div/span[1]')
more_button.click()

# 구독자 일일 차트에 마우스 호버 실행
element_to_hover_over = driver.find_element_by_xpath('//*[@id="channel-history-sub-chart"]/div[1]/canvas')
hover = ActionChains(driver).move_to_element(element_to_hover_over) # 차트의 가운데로 마우스 호버
hover.perform()

# 차트 사이즈 가져오기
element_size = element_to_hover_over.size
element_height = element_size['height']
element_width = element_size['width']

# 차트의 좌측으로 이동
x_offset = (element_width / 2)
move = ActionChains(driver).move_by_offset(-x_offset, 0)
move.perform()

# 차트의 우측으로 이동하면서 날짜/일별 구독자수 저장
list_daily_sub_date = []
list_daily_subscribe_cnt = []
for idx in range(int(element_width / 2)):
    # 마우스 이동
    move = ActionChains(driver).move_by_offset(2, 0)
    move.perform()
    
    chart_info = driver.find_element_by_xpath('//*[@id="channel-history-sub-chart"]/div[2]').text
    if chart_info.find('\n') != -1 :
        list_daily_sub_date.append(chart_info.split('\n')[0])
        subscribe_cnt = float(chart_info.split('\n')[1])
        list_daily_subscribe_cnt.append(subscribe_cnt)
'''        
# page down scroll
body = driver.find_element_by_tag_name('body')
num_of_pagedowns = 1
while num_of_pagedowns:
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(1.5)
    num_of_pagedowns -= 1
    
# 일별 조회수 차트에 마우스 호버 실행
element_to_hover_over = driver.find_element_by_xpath('//*[@id="channel-history-view-chart"]/div[1]/canvas')
hover = ActionChains(driver).move_to_element(element_to_hover_over) # 차트의 가운데로 마우스 호버
hover.perform()

# 차트 사이즈 가져오기
element_size = element_to_hover_over.size
element_height = element_size['height']
element_width = element_size['width']

# 차트의 좌측으로 이동
x_offset = (element_width / 2)
move = ActionChains(driver).move_by_offset(-x_offset, 0)
move.perform()

# 차트의 우측으로 이동하면서 날짜/일별 조회수 저장
list_view_date = []
list_view_cnt = []
for idx in range(int(element_width / 2)):
    # 마우스 이동
    move = ActionChains(driver).move_by_offset(2, 0)
    move.perform()
    
    chart_info = driver.find_element_by_xpath('//*[@id="channel-history-view-chart"]/div[2]').text
    
    find_idx = chart_info.find('\n') # 차트 예외 검사
    
    if find_idx != -1 : # 차트 예외 처리(날짜 \n 조회수 포맷이 아닌경우)
        find_idx_over = chart_info.find('\n', find_idx) # 차트 예외 검사
        if find_idx_over == -1: # 차트 예외 처리(날짜 \n 조회수 포맷이 맞을 경우)
            list_view_date.append(chart_info.split('\n')[0])
            
            view_cnt = chart_info.split('\n')[1]
            if view_cnt.find('만') != -1:
                view_cnt = view_cnt.replace('만', '')
                view_cnt = float(view_cnt) * 10000
            elif view_cnt.find('천') != -1:
                view_cnt = view_cnt.replace('천', '')
                view_cnt = view_cnt * 1000
            else:
                view_cnt = float(view_cnt)
        list_view_cnt.append(view_cnt)

driver.close()

'''
# 금일 데이터 삭제
del list_sub_date[-1]
del list_subscribe_cnt[-1]
del list_daliy_sub_date[-1]
del list_daliy_subscribe_cnt[-1]
del list_view_date[-1]
del list_view_cnt[-1]
'''

# create xlsx file
df_channel_sub = pd.DataFrame(columns = {'date', 'subscribe count'})  
df_channel_sub['date'] = list_sub_date
df_channel_sub['subscribe count'] = list_subscribe_cnt
df_channel_sub = df_channel_sub.drop_duplicates(['date'], keep='first') # 중복제거
df_channel_sub = df_channel_sub.reset_index(drop=True)



list_sub_cnt = [int(x) for x in df_channel_sub['subscribe count']]
list_daily_subscribe_cnt = []
for idx in range(len(list_sub_cnt) - 1):   
    daily_subscribe_cnt = list_sub_cnt[idx + 1] - list_sub_cnt[idx]
    daily_subscribe_cnt = round(daily_subscribe_cnt, -1)
    list_daily_subscribe_cnt.append(int(daily_subscribe_cnt))
    
df_channel_sub['daily subscribe count'] = pd.Series(list_daily_subscribe_cnt)
df_channel_sub = df_channel_sub[df_channel_sub['daily subscribe count'].notna()] # nan 행 제거

df_channel_daily_view = pd.DataFrame(columns = {'date', 'daily view count'})        
df_channel_daily_view['date'] = list_view_date
df_channel_daily_view['daily view count'] = list_view_cnt
df_channel_daily_view = df_channel_daily_view.drop_duplicates(['date'])# 중복제거

df_channel_info = pd.merge(df_channel_sub, df_channel_daily_view, on = 'date')
# change dataframe columns order
df_channel_info = df_channel_info[['date', 'subscribe count', 'daily subscribe count', 'daily view count']]

filename = username + '_channel_info.xlsx'
df_channel_info.to_excel(filename)







