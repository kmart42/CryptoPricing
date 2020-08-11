import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing, svm
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression

from sklearn import linear_model
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline

import matplotlib
import matplotlib.pyplot as plt

path_all = 'C:\\Users\\Captain\\Crypto\\Data\\bitcoin_new.csv'
# df = pd.read_csv(path_all, sep=',', index_col=['Date'])
df = pd.read_csv(path_all,
                 usecols=["Date", "Price", 'Open', 'High', 'Low', 'Vol.', 'Change %'],
                 parse_dates=True,
                 index_col=["Date"])

df['HL'] = (df['High'] - df['Low']) / (df['Low'] * 100)
for_col = 'Price'
forecast = int(30)
df['label'] = df[for_col].shift(-forecast)
# df['label'] = df[for_col]

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_forecast_out = X[-forecast:]
X = X[:-forecast]
y = np.array(df['label'])
y = y[:-forecast]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ridge_def = linear_model.Ridge(alpha=200)
ridge_def.fit(X_train, y_train)
# Test
accuracy = ridge_def.score(X_test, y_test)
print("Accuracy of Ridge Regression: ", accuracy)

for_out = ridge_def.predict(X_forecast_out)

Xnew, _ = make_regression(n_samples=30, n_features=len(df.columns) - 1, noise=0.1, random_state=1)
Xnew = np.array(Xnew)
Xnew = preprocessing.scale(Xnew)
print(Xnew)
ridge_out = ridge_def.predict(Xnew)
#
#
predicted = np.array([])
for t in range(0, 30):
    predicted = np.append(predicted, ridge_out[t])

print(predicted)

# df.dropna(inplace=True)
# df['Predicted'] = np.nan
# last_date1 = df.iloc[-1].name
# last_unix1 = last_date1.timestamp()
# one_day1 = 86400
# next_unix1 = last_unix1 + one_day1
#
# for t in ridge_out:
#     next_date1 = datetime.datetime.fromtimestamp(next_unix1)
#     next_unix1 += 86400
#     df.loc[next_date1] = [np.nan for _ in range(len(df.columns)-1)]+[t]

df.dropna(inplace=True)
df['forecast'] = np.nan
# last_date = df.iloc[-1].name + pd.Timedelta(forecast, unit='D')
last_date = df.iloc[-1].name
# print(last_date)
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in for_out:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Price'].plot(figsize=(15, 6), color="green")
df['forecast'].plot(figsize=(15, 6), color="orange")
# df['Predicted'].plot(figsize=(15, 6), color="purple")
plt.xlim(xmin=datetime.date(2020, 6, 1))
plt.xlim(xmax=datetime.date(2020, 9, 1))
plt.ylim(ymin=500)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
# plt.show()

# out_data = [df.forecast]
# df_out = pd.DataFrame(out_data)
# df_out = df_out.T
# df_out.to_csv('./ridge_test3.csv')
