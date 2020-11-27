import finnhub
import time
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

finnhub_client = finnhub.Client(api_key="buvb4kf48v6vrjludb90")

WINDOW = 1000
d = {'price': []}
df = pd.DataFrame(data=d)
df2 = df.copy(deep=False)
forecast_out = int(5)
total = WINDOW
test = .2 * total
train = total - test
buff = 0
regr = Ridge(alpha=200)

# for res in range(100):
#     df = df.append({'price': res}, ignore_index=True)
#     if df.shape[0] > WINDOW:
#         start_time = time.time()
#         df = df.tail(WINDOW)
#         df['Prediction'] = df[['price']].shift(-forecast_out)
#         X = np.array(df.drop(['Prediction'], 1))
#         X = preprocessing.scale(X)
#         X_forecast = X[-forecast_out:]
#         X = X[:-forecast_out] # remove forcast from X
#         y = np.array(df['Prediction'])
#         y = y[:-forecast_out]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#         # Training
#         clf = regr
#         clf.fit(X_train, y_train)
#         # Testing
#         confidence = clf.score(X_test, y_test)
#         print("confidence: ", confidence)
#         forecast_prediction = clf.predict(X_forecast)
#         print(forecast_prediction)
#         print("--- %s seconds ---" % (time.time() - start_time))
#         df2 = df2.append({'price': res, 'conf': confidence, 'pred': forecast_prediction}, ignore_index=True)
#         # df2 = df2.append({'conf': confidence}, ignore_index=True)
#         # df2 = df2.append({'pred': forecast_prediction}, ignore_index=True)
#         # time.sleep(.5)



# for res in range(20):
#     df = df.append({'price': res}, ignore_index=True)
#     if df.shape[0] > WINDOW:
#         df = df.tail(WINDOW)
#         time_data = df['price']
#         y_train = time_data[buff:train]
#         y_test = time_data[train:total]
#         x_train = pd.DataFrame([list(time_data[i:i + buff]) for i in range(train - buff)],
#                                columns=range(buff, 0, -1), index=y_train.index)
#         x_test = pd.DataFrame([list(time_data[i:i + buff]) for i in range(train - buff, total - buff)],
#                               columns=range(buff, 0, -1), index=y_test.index)



# while True:
i = 0
print('***BUILDING DATAFRAME***')
while i < 20:
    res = finnhub_client.quote('BINANCE:BTCUSDT')['c']
    df = df.append({'price': res}, ignore_index=True)
    if df.shape[0] > WINDOW:
        start_time = time.time()
        df = df.tail(WINDOW)
        df['Prediction'] = df[['price']].shift(-forecast_out)
        X = np.array(df.drop(['Prediction'], 1))
        X = preprocessing.scale(X)
        X_forecast = X[-forecast_out:]
        X = X[:-forecast_out] # remove forcast from X
        y = np.array(df['Prediction'])
        y = y[:-forecast_out]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Training
        clf = regr
        clf.fit(X_train, y_train)
        # Testing
        confidence = clf.score(X_test, y_test)
        print("confidence: ", confidence)
        forecast_prediction = clf.predict(X_forecast)
        print(forecast_prediction)
        print("--- %s seconds ---" % (time.time() - start_time))
        df2 = df2.append({'price': res, 'conf': confidence, 'pred': forecast_prediction}, ignore_index=True)
        i += 1
    time.sleep(1)
df2.to_csv(r'./df_test.csv', index=False, header=True)
plt.plot(df2.tail(3))
plt.show()
