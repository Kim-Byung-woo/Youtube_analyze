# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 21:52:35 2020

@author: user
"""


#%%
from selenium.webdriver import Chrome
import time
import sqlite3
from pandas.io import sql
import os
import pandas as pd
from selenium import webdriver

# select box option 개수 구하기 모듈 추가 import
from selenium.webdriver.support.ui import Select

#%%
options = webdriver.ChromeOptions()
#options.add_argument("--start-maximized");
browser = webdriver.Chrome('chromedriver', options=options)
browser.get('http://stat.suwon.go.kr/stat/stats/statsView.do?categorySeqNo=16')
browser.implicitly_wait(5)

# 연도 select box option 개수 구하기
sb_year = Select(browser.find_element_by_id("baseYearSelect"))
options_year = sb_year.options
options_year_cnt = len(options_year)


for i in range(options_year_cnt): # 연도 select box list 개수 만큼 반복문 실행
    index_year = i + 1 # 1부터 시작
    browser.find_element_by_xpath(f"//*[@id='baseYearSelect']/option[{index_year}]").click() # 연도 select box 클릭

    # 월말 select box option 개수 구하기
    sb_month = Select(browser.find_element_by_id("baseSelect"))
    options_month = sb_month.options
    options_month_cnt = len(options_month)
    
    for j in range(options_month_cnt): # 월말 select box list 개수 만큼 반복문 실행
        index_month = j + 1 # 1부터 시작
        browser.find_element_by_xpath(f"//*[@id='baseSelect']/option[{index_month}]").click() # 월말 select box 클릭
        browser.implicitly_wait(5)
        browser.find_element_by_xpath('//*[@id="baseSelectGo"]').click() # go 버튼
        browser.implicitly_wait(5)
        # browser.find_element_by_xpath('//*[@id="container_in2"]/div[4]/div[3]/a/img').click() # 다운로드 버튼
        time.sleep(2)

browser.close() # webdriver 종료
