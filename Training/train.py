import pandas as pd

path_daily = 'C:\\Users\\Captain\\Crypto\\Data\\crypto.csv'
path_all = 'C:\\Users\\Captain\\Crypto\\Data\\bitcoin_all.csv'
# df = pd.read_csv(path_daily, sep=',')
df = pd.read_csv(path_all, sep=',')
# del_col_list = ['site_url', 'github_url', 'platform_name', 'industry_name', 'crypto_type']
# df.drop(del_col_list, axis=1, inplace=True)
# df.dropna(inplace=True)
# split = .3
# days = 1200
# tick = 'BTC'

# btc_data = df[df['ticker'] == tick]
# time_data = btc_data['price_usd']
time_data = df['Price']

# total = df[df.ticker == tick].shape[0]
total = df.shape[0]
# test = total - 30
# test2 = int(round(total * .1))
test = 90
# print(test)
train = total - test
buff = 80
# prev_values = time_data.iloc[:days]
y_train = time_data[buff:train]
y_test = time_data[train:total - 30]
x_train = pd.DataFrame([list(time_data[i:i + buff]) for i in range(train - buff)],
                       columns=range(buff, 0, -1), index=y_train.index)
x_test = pd.DataFrame([list(time_data[i:i + buff]) for i in range(train - buff, total - buff - 30)],
                      columns=range(buff, 0, -1), index=y_test.index)

# x_new = pd.DataFrame([list(time_data[i:i + buff]) for i in range(total - buff - 30, total - buff)],
#                     columns=range(buff, 0, -1))

# x_new = pd.DataFrame(list(df['Price'][total - 30:total]), columns=range(buff, 0, -1), index=y_test.index)

x_new = df[-30:]


def evaluate(model):
    stats_mod = model.fit(x_train, y_train)
    pred_mod = model.predict(x_test)
    fut_mod = model.predict(x_new)
    return pred_mod, stats_mod, fut_mod
