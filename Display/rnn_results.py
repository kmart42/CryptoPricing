import matplotlib.pyplot as plt
import joblib
import pandas as pd

df = pd.read_csv('../Data/bitcoin_new.csv', sep=',')
df = df.set_index("Date")[['Price']].tail(2000)
df = df.set_index(pd.to_datetime(df.index))

loc = '../Models/'
n = '_9'
suf = '.sav'

mod_name = loc + 'rnn' + n + suf
y_name = loc + 'yhat' + n + suf
act = loc + 'actual' + n + suf
act2 = loc + 'actual2' + n + suf
pred_name = loc + 'preds' + n + suf

model = joblib.load(mod_name)
yhat = joblib.load(y_name)
actual = joblib.load(act)
actual2 = joblib.load(act2)
preds = joblib.load(pred_name)

# Plotting
plt.figure(figsize=(16, 6))
plt.plot(actual2, label="Actual")
plt.plot(preds, label="Predicted")
plt.plot(yhat, label='Forecast')
plt.ylabel("Price")
plt.xlabel("Dates")
plt.title(f"Forecasting the next {len(yhat)} days")
plt.legend()
plt.show()

# print_data = [df.index, actual2.Price, preds.Price, yhat.Price]
# print_df = pd.DataFrame(print_data)
# print_df = print_df.T
# print_df.to_csv('./rnn_graph.csv')
