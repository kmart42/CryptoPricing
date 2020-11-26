import finnhub
import time
import pandas as pd
from sklearn.linear_model import Ridge

finnhub_client = finnhub.Client(api_key="buvb4kf48v6vrjludb90")

WINDOW = 100
d = {'price': []}
df = pd.DataFrame(data=d)
total = WINDOW
test = 9
train = total - test
buff = 0
regr = Ridge(alpha=200)


# for res in range(20):
#     df = df.append({'price': res}, ignore_index=True)
#     df = df.tail(2)
#     print(df['price'])
# print(df)


while True:
    res = finnhub_client.quote('BINANCE:BTCUSDT')['c']
    df = df.append({'price': res}, ignore_index=True)
    if df.shape[0] > WINDOW:
        df = df.tail(WINDOW)
        time_data = df['price']
        # y_train = time_data[buff:train]
        # y_test = time_data[train:total - 30]
        # x_train = pd.DataFrame([list(time_data[i:i + buff]) for i in range(train - buff)],
        #                        columns=range(buff, 0, -1), index=y_train.index)
        # x_test = pd.DataFrame([list(time_data[i:i + buff]) for i in range(train - buff, total - buff - 30)],
        #                       columns=range(buff, 0, -1), index=y_test.index)
        print(df['price'])
    time.sleep(.5)
