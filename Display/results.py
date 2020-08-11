import pandas as pd
from Training.models import ridge_mod, ridge_stats, ridge_fut, svr_mod, svr_stats, svr_fut
from Display.graph import print_stats, plot_pred
from Training.train import x_test, y_test, y_train, test, train, buff, total, df

# print_stats(ridge_stats, ridge_mod, x_test, y_test)
# plot_pred(ridge_mod, y_train, 'Ridge')
dates = []
actual = []
forecast = []
future = []


for t in range(total - test, total - 30):
    dates.append(df.Date[t])
    actual.append(y_test[t])
    forecast.append(ridge_mod[t - total + test])
    future.append(0)

for i in range(0, 30):
    future.append(ridge_fut[i])
    dates.append(df.Date[i + total - 30])


ridge_data = [dates, actual, forecast, future]
ridge_d = pd.DataFrame(ridge_data)
ridge_d = ridge_d.T
ridge_d.to_csv('./ridge_graph2.csv')
