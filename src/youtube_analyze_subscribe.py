# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 04:54:59 2020

@author: user
"""


#%%
# 누적 구독자 변화 예측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

def normalized_df(price_df, seq_len):    
    sequence_length = seq_len + 1 # seq_len + 1 -> 최근 seq_len 일만큼의 데이터를 보고 내일 것을 예측을 해야하기 때문에( 즉 seq_len이 50이면, 50개를 보고 1개를 예측을 해야하기 때문에 윈도우는 총 51개 입니다.)
    
    # 데이터 전처리 - 정규화(첫날을 기준)
    result = []
    for index in range(len(price_df) - sequence_length):
        result.append(price_df[index: index + sequence_length])
    normalized_data = []
    for window in result:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    
    result = np.array(normalized_data)
    
    X = result[:, :-1]
    Y = result[:, -1]
    return X, Y


# 미래 예측
def predict_subscribe(list_subscribe, seq_len):
    # X[-1] -> 금일 기준으로 seq_len만큼의 전일 데이터
    x_today = X[-1].reshape((1, seq_len, 1)) # predict를 하기위해서 차원 변화
    # 예측값 출력
    y_today = model.predict(x_today) # 훈련된 모델 사용
    y_today = np.array(y_today).flatten().tolist() # 2차원 배열 -> 1차원
    
    # 측정된 예측값을 역보정(역정규화)하여 실제 값하고 비교해봅니다.
    standard_subscribe = list_subscribe[len(list_subscribe) - seq_len] # 기준 종가  
    # 역보정(역정규화)
    today_subscribe = (y_today[0] + 1) * standard_subscribe

    return today_subscribe
 
#%%
# 데이터 로드
# load file
xlxs_dir = '이스타TV_channel_info_week.xlsx'
df_read_channel_info = pd.read_excel(xlxs_dir)
df_read_channel_info = df_read_channel_info.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제


subscribe = list(df_read_channel_info['subscribe count'].values) # 누적 구독자 데이터 추출
seq_len = 50 # 기준 날짜 설정(50 -> 최근 50일)
# 데이터 분리 및 정규화
X, Y = normalized_df(subscribe, seq_len)

# 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.1, 
                                                    random_state = 0)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 모델 구성
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(seq_len, 1))) # 왜 input_shape를 2차원(reshape는 3차원인데)? -> input_shape는 시퀀스의 크기(X_train.shape[1], 1)를 지정합니다.
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')

model.fit(X_train, Y_train,validation_data=(X_test, Y_test),batch_size=1,epochs=10)

# 예측 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))

from sklearn.metrics import r2_score
print('결정계수',r2_score(Y_test, Y_prediction))

# 시각화
pred = model.predict(X_test)

# %matplotlib inline
plt.rc('font', family='gulim') 
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(Y_test, label='실값')
ax.plot(pred, label='예측값')
ax.legend() # 범주

#%%
# 최초 1회 예측 실행
list_predict = list()
subscribe = list(df_read_channel_info['subscribe count'].values) # 종가 데이터 추출

# 2018.10.30일 예측값이 무엇? -> model.predict에 2018.10.29의 50일전 부터 2018.10.29까지 데이터를 넣으면 2018.10.30일 예측값이 나옵니다.
# X[-1] -> 2018.10.30 50일 전 데이터
x_today = X[-1].reshape((1, seq_len, 1)) # predict를 하기위해서 차원 변화
# 예측값(2018.10.30일 예측값) 출력
y_today = model.predict(x_today) # 훈련된 모델 사용

y_today = np.array(y_today).flatten().tolist()

# 나온 예측값을 역보정(역정규화)하여 실제 값하고 비교해봅니다.
standard_subscribe = subscribe[len(subscribe) - seq_len] # 기준 종가  
# 역보정(역정규화)
today_subscribe = (y_today[0] + 1) * standard_subscribe

subscribe.append(round(today_subscribe, -1)) # 초기 데이터셋(종가 데이터)에 예측한 결과 누적
list_predict.append(round(today_subscribe, -1))

# 설정된 기간 만큼 예측 실행
date_cnt = 7
date_cnt = date_cnt - 1
for idx in range(date_cnt):
    X, Y = normalized_df(subscribe, seq_len) # 누적된 결과를 다시 정규화 (다음날 예측을 위해서)
    today_subscribe = predict_subscribe(subscribe, seq_len)
    subscribe.append(round(today_subscribe, -1)) # 초기 데이터셋(종가 데이터)에 예측한 결과 누적
    list_predict.append(round(today_subscribe, -1))

# 실제값
xlxs_dir = '이스타TV_channel_info_old.xlsx'
df_read_channel_info_old = pd.read_excel(xlxs_dir)
df_read_channel_info_old = df_read_channel_info_old.iloc[:, 1:] # excel 파일에 에서 가져온 index colum 삭제
list_real = list(df_read_channel_info_old.iloc[len(df_read_channel_info_old) - (date_cnt + 1):, :]['subscribe count'])

# 예측값
plt.rc('font', family='gulim') 
fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(111)
ax.plot(pd.Series(list_real), label='실제 누적 구독자')
ax.plot(pd.Series(list_predict), label='예측 누적 구독자')
ax.legend() # 범주
#file_path = Finance_code + '_' +start_date + '_' + str(date_cnt) + '.png'
#plt.savefig(file_path, dpi=50)