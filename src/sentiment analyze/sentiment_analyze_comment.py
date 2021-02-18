#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

file_dir = os.getcwd() # 현재 파일 경로 추출

'''
현재 파일 경로가 src/sentiment analyze/에 위치하기 때문에 
data 파일 경로를 알기 위해서는 상위 경로 추출을 2번 실행 합니다.
'''
file_dir = os.path.dirname(file_dir) # 상위 경로 추출
file_dir = os.path.dirname(file_dir) # 상위 경로 추출
#%%
# 크롤링한 댓글 불러오기

xlxs_dir = file_dir + '/data/comment_train.xlsx'
train_data = pd.read_excel(xlxs_dir)

xlxs_dir = file_dir + '/data/comment_test.xlsx'
test_data = pd.read_excel(xlxs_dir)
#%%
# 데이터 개수 확인
print('훈련 데이터 리뷰 개수 :',len(train_data)) # 리뷰 개수 출력
print('테스트 데이터 리뷰 개수 :',len(test_data)) # 리뷰 개수 출력
# %%
# 텍스트 전처리
# 훈련용 리뷰 데이터 중 중복값 제거
train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
# 훈련용 리뷰 데이터 중 한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 훈련용 리뷰 데이터 중 모두 제거된 데이터는 Nan값으로 대체
train_data['document'].replace('', np.nan, inplace=True)
# 훈련용 리뷰 데이터 중 Nan값 제거
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인
print('전처리 후 훈련용 데이터 개수: ', len(train_data))

# 테스트용 리뷰 데이터 중 중복값 제거
test_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
#테스트용  리뷰 데이터 중 한글과 공백을 제외하고 모두 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
#테스트용  리뷰 데이터 중 모두 제거된 데이터는 Nan값으로 대체
test_data['document'].replace('', np.nan, inplace=True)
#테스트용  리뷰 데이터 중 Nan값 제거
test_data = test_data.dropna(how = 'any') # Nan 값이 존재하는 행 제거
print(test_data.isnull().values.any()) # Nan 값이 존재하는지 확인
print('전처리 후 테스트용 데이터 개수: ', len(test_data))
#%%
# 토큰화
# 토큰(Token)이란 문법적으로 더 이상 나눌 수 없는 언어요소를 뜻합니다. 텍스트 토큰화(Text Tokenization)란 말뭉치(Corpus)로부터 토큰을 분리하는 작업을 뜻합니다.
from konlpy.tag import *
import json
okt = Okt() 
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# 기존에 태깅한 데이터가 있으면 불러옵니다.
if os.path.isfile(file_dir + "/data/train_comment.json"):
    print('Json File is already')
    with open(file_dir + "/data/train_comment.json", encoding='UTF8') as f:
        X_train = json.load(f)
    with open(file_dir + "/data/test_comment.json", encoding='UTF8') as f:
        X_test = json.load(f)
else: # 태깅한 데이터 없는 경우 태깅 실시
    X_train = [] 
    for sentence in train_data['document']: 
        temp_X = [] 
        temp_X = okt.morphs(sentence, stem=True) # 토큰화 
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
        X_train.append(temp_X)
    
    X_test = [] 
    for sentence in test_data['document']: 
        temp_X = [] 
        temp_X = okt.morphs(sentence, stem=True) # 토큰화 
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
        X_test.append(temp_X)
    
    # JSON 파일로 저장
    with open(file_dir + "/data/train_comment.json", 'w', encoding="utf-8") as make_file:
        json.dump(X_train, make_file, ensure_ascii=False, indent="\t")
    with open(file_dir + "/data/test_comment.json", 'w', encoding="utf-8") as make_file:
        json.dump(X_test, make_file, ensure_ascii=False, indent="\t")
        
#%%
# 토큰화 된 단어 정수 인코딩 - 텍스트를 숫자로 처리할 수 있도록
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index) # 총 단어가 43000개 넘게 존재
#%%
# 단어 등장 빈도수가 3회 미만인 단어의 비중을 확인합니다.
threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value # 빈도수(value)를 카운팅

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# 희귀 단어 = 등장 빈도수가 threshold 보다 작은 단어
print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100) 

# 전체 단어 개수 중 빈도수 3미만인 단어 개수는 제거.
# 단어가 43000개 중 상위 희귀단어(빈도가 3미만)는 제거가 되어 문장에서 희귀단어는 OOV(Out of Vocabulary )로 처리가 됩니다.
# ex. 빈도수가 높은 상위단어 개수가 5개이면 vocab_size(5) + 2 합니다
# +1은 인덱스의 시작이 1이라서, +1은 oov의 인덱스를 1로 설정하기 위해서 따라서 상위 빈도수 단어를 사용하기 위해서는 vocab_size + 2 합니다
vocab_size = total_cnt - rare_cnt + 2 # 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
print('단어 집합의 크기 :',vocab_size)

# 토큰화 된 단어 정수 인코딩 - 텍스트를 숫자로 처리할 수 있도록
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index)

X_train = tokenizer.texts_to_sequences(X_train) # X_train안에 텍스트 데이터를 숫자의 시퀀스 형태로 변환합니다.
X_test = tokenizer.texts_to_sequences(X_test) # X_test 텍스트 데이터를 숫자의 시퀀스 형태로 변환합니다.

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])
#%%
# 희귀 단어들로만 이루어진 샘플들 제거
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1] # enumerate를 활용해서 길이가 1보다 작은 샘플의 인덱스를 저장합니다.

# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))
#%%
# 패딩: 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업
print('댓글의 최대 길이 :',max(len(l) for l in X_train))
print('댓글의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len = 70 # movie review: 35, comment: 70
def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

below_threshold_len(max_len, X_train)

# 케라스 전처리 도구로 패딩하기
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
# %%
# 모델 생성 및 훈련
# LSTM 모델로 영화 리부 감성 분류
from keras.layers import Embedding, Dense, LSTM 
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model_dir = file_dir + '/data/comment_model_wiki.h5'
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint(model_dir, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# 모델 훈련
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
#%%
# 모델 검증
from tensorflow.keras.models import load_model
loaded_model = load_model(model_dir)
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
# %%
list_accuracy = []

def sentiment_predict(new_sentence):
  new_morphs = okt.morphs(new_sentence, stem=True) # 토큰화
  new_morphs = [word for word in new_morphs if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_morphs]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print(new_sentence, end = '')
    print("는 {:.2f}% 확률로 긍정 댓글입니다.\n".format(score * 100))
    label_okt.append(1)
    accuracy = score * 100
    accuracy = round(accuracy, 2)
    list_accuracy.append(accuracy)
  else :
    print(new_sentence, end = '')
    print("는{:.2f}% 확률로 부정 댓글입니다.\n".format((1 - score) * 100))
    label_okt.append(0)
    accuracy = (1 - score) * 100 
    accuracy = round(accuracy, 2)
    list_accuracy.append(accuracy)
#%%
# 데이터셋 로드
xlxs_dir = file_dir + '/data/엘론_video_info.xlsx'
df_comment = pd.read_excel(xlxs_dir, sheet_name = 'comment')

# 데이터 개수 확인
print('댓글 개수 :',len(df_comment)) # 리뷰 개수 출력
# %%
# 텍스트 전처리
# 댓글 중 중복값 제거
df_comment.drop_duplicates(subset=['comment'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
# 댓글 중 한글과 공백을 제외하고 모두 제거
df_comment['comment'] = df_comment['comment'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 댓글 중 공백만 있는 경우 제거
df_comment['comment'] = df_comment['comment'].str.strip() 
# 댓글 중 모두 제거된 데이터는 Nan값으로 대체
df_comment['comment'].replace('', np.nan, inplace=True)

# 훈련용 리뷰 데이터 중 Nan값 제거
df_comment = df_comment.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(df_comment.isnull().values.any()) # Null 값이 존재하는지 확인
print('전처리 후 댓글 개수: ', len(df_comment))

df_comment.reset_index(inplace = True) # 행제거 인덱스도 같이 삭제되어 for문을 돌리기 위해서 인덱스 컬럼 초기화
df_comment = df_comment[['comment id', 'comment']] # 기존 인덱스 컬럼 삭제
#%%
label_okt = []
list_accuracy = []

for idx in range(len(df_comment)):
    sentece = df_comment['comment'][idx]
    sentiment_predict(sentece)

df_comment['label'] = label_okt
df_comment['accuracy'] = list_accuracy

# 감정분석 결과 저장
df_comment.to_excel(file_dir + '/data/comment_base_result_wiki' +'.xlsx')




























# %%
