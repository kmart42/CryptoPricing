import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def plot_pred(pred_model, y_act, title):
    plt.figure(figsize=(17, 5))
    # plt.plot(prev_values, label='Train')
    plt.plot(y_act, label='Actual')
    plt.plot(pred_model, label='Predicted')
    plt.ylabel("Price")
    plt.title(title, fontsize=20)
    plt.legend()
    plt.show()


def mape(tr, pred):
    tr, pred = np.array(tr), np.array(pred)
    return np.mean(np.abs((tr - pred) / tr)) * 100


def print_stats(mod_stat, pred, test_x, test_y):
    print('Accuracy: ', mod_stat.score(test_x, test_y))
    print('MSE: ', "%.2f" % mse(pred, test_y))
    print('MAE: ', "%.2f" % mae(pred, test_y))
    print('MAPE: ', "%.2f" % mape(pred, test_y))
