# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:06:09 2020

@author: A
"""
#%%
'''
============== Read me ================

video_url 변수에 크롤링 하고 싶은 유튜브 영상의 주소를 저장 후 프로그램을 실행해야 합니다.
1. https://www.youtube.com에 점속
2. 사용자가 원하는 채널을 클릭합니다.
3. 해당 채널 동영상 버튼 클릭
4. 해당 채널 동영상 탭 주소 복사

5. url에 복사된 동영상 탭 주소를 저장.
6. 프로그램(크콜링) 실행 -자동으로 크롬 드라이버 실행되오니 드라이버를 강제 종료 하지 마시오
6-1. 동영상 1개당 크롤링 시간이 2초이상 걸립니다. 따라서 상당히 오래걸리니 시간적 여유가 있을 때 실행하기 바랍니다.
    (실행된 크롬 드라이버를 종료하지 않은 상태에서 다른 작업 가능합니다.)
7. 저장된 xlsx 파일 확인 - 영상길이 데이터 추가 예정

'''
#%%
def str_to_date(string):
    
    yyyymmdd = string
    
    yyyy= yyyymmdd.split('-')[0]
    mm= yyyymmdd.split('-')[1]
    dd= yyyymmdd.split('-')[2]
    
    if int(mm) < 10 :
        mm = '0'+mm
           
    if int(dd) < 10 :
        dd = '0'+dd                                                       
                                  
    return yyyy + '-' + mm + '-' + dd       
#%%

import numpy as np
import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

driver = webdriver.Chrome('chromedriver.exe')

# 크리에이터의 video 전체가 나오는 페이지를 연다.
url = 'https://www.youtube.com/c/%EC%9D%B4%EC%8A%A4%ED%83%80TV/videos'
driver.get(url)

#2. 크롤링을 위해 화면 맨 아래까지 스크롤 내리기
SCROLL_PAUSE_TIME = 0.5# 한번 스크롤 하고 멈출 시간 설정

body = driver.find_element_by_tag_name('body')# body태그를 선택하여 body에 넣음

# 동영상 페이지 제일 밑으로 스크롤
while True:
    last_height = driver.execute_script('return document.documentElement.scrollHeight')
    # 현재 화면의 길이를 리턴 받아 last_height에 넣음
    for i in range(10):
        body.send_keys(Keys.END)
        # body 본문에 END키를 입력(스크롤내림)
        time.sleep(SCROLL_PAUSE_TIME)
    new_height = driver.execute_script('return document.documentElement.scrollHeight')
    if new_height == last_height:
        break;
        
page = driver.page_source
soup = BeautifulSoup(page, 'lxml')

# 채널명 크롤링
username = soup.find('div', {'id' : 'text-container'}).text
username= username.strip()

# 제목, url 크롤링
all_videos = soup.find_all(id='dismissable')

list_title = []
list_url = []
list_video_length = []
for video in all_videos:
    title = video.find(id='video-title')
    if len(title.text.strip())>0: # 공백을 제거하고 글자수가 0보다 크면 append 
        list_title.append(title.text)
    
    #find('a',{'id':'thumbnail'})['href']
    url = video.find(id='video-title')['href'] # url append
    list_url.append(url)

    video_lenth = video.find('span',{'class' : 'style-scope ytd-thumbnail-overlay-time-status-renderer'})
    list_video_length.append(video_lenth.text.strip())

    '''
    if(len(list_url)) >= 600: # url 최대 개수 설정(시간관계상)
        break
    '''

# 영상 업로드 날짜 크롤링
list_upload_date = []
list_view_count = []
for url in list_url:
    video_url = 'https://www.youtube.com' + url

    driver.get(video_url)
        #driver.maximize_window()
    body = driver.find_element_by_tag_name('body')
    time.sleep(1.5)
    
    page = driver.page_source
    soup = BeautifulSoup(page, 'lxml')

    # 영상 업로드 날짜, 조회수 추출
    '''
    날짜 데이터 정규식으로 추출하기 
    ex) 최초공개 2017.10.10 처럼 들어오는 경우 있음 -> 날짜부분만 정규식으로 추출하기
    '''
    upload_date = soup.find('div', {'id' : 'date'}).find('yt-formatted-string', 'style-scope ytd-video-primary-info-renderer').text
    upload_date = upload_date.replace('. ', '-') # 날짜 포맷 수정
    upload_date = upload_date.replace('.', '') # 날짜 포맷 수정
    upload_date = str_to_date(upload_date)
    list_upload_date.append(upload_date)
    
    view_cnt = soup.find('span', 'view-count style-scope yt-view-count-renderer').text
    view_cnt = int(''.join(ele for ele in view_cnt if ele.isdigit() or ele == '.')) # 문자열에서 숫자 추출
    list_view_count.append(view_cnt)
driver.close()   

# create xlsx file
df_video_info = pd.DataFrame({'upload_date':[]})
df_video_info['title'] = list_title
df_video_info['running time'] = list_video_length
df_video_info['upload_date'] = list_upload_date
df_video_info['view count'] =  list_view_count


df_upload_cnt = pd.DataFrame()
df_upload_cnt = df_video_info.groupby("upload_date").count()
df_upload_cnt = df_upload_cnt.iloc[: , : 1]
df_upload_cnt.columns = ["daily video count"]

xlxs_dir = username + '_upload_cnt.xlsx'
# xlsx 파일 생성
# Write two DataFrames to Excel using to_excel(). Need to specify an ExcelWriter object first.
# 2개 이상의 DataFrame을 하나의 엑셀 파일에 여러개의 Sheets 로 나누어서 쓰려면 먼저 pd.ExcelWriter() 객체를 지정한 후에, sheet_name 을 나누어서 지정하여 써주어야 합니다. 
with pd.ExcelWriter(xlxs_dir) as writer:
    df_video_info.to_excel(writer, sheet_name = 'video info')
    df_upload_cnt.to_excel(writer, sheet_name = 'upload count')                 
                              
                              
                              
                              
                                              
              
                              
                              
                              
                              
                              
                              
                              
                              