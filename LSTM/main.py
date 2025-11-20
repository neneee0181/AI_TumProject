import yfinance as yf
import pandas as pd

# 데이터 불러오기
def load_stock(code="005930.KS", period="10y"):
    df = yf.download(code, period=period)
    df = df[['Close']]  # 종가만 사용
    df.dropna(inplace=True)
    return df

df = load_stock()
print(df.head())

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
from utils import create_dataset

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Close']])

seq_len = 120
X, y = create_dataset(scaled, seq_len)

# train/test split (80:20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 텐서 변환
import torch

X_train = torch.tensor(X_train).float()
X_test  = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test  = torch.tensor(y_test).float()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# 모델 학습 및 예측
from train import train_model
from predict import predict_and_plot

model = train_model(X_train, y_train, epochs=200)
predict_and_plot(model, X_test, y_test.numpy(), scaler)
