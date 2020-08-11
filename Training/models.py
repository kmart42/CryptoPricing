from sklearn import linear_model
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from Training.train import evaluate
ticker = "BTC"


# Ridge regression
ridge_def = linear_model.Ridge(alpha=200)
ridge_mod, ridge_stats, ridge_fut = evaluate(ridge_def)

print(ridge_fut)

# Linear SVR
svr_def = make_pipeline(LinearSVR())
svr_mod, svr_stats, svr_fut = evaluate(svr_def)

# TODO run SNN on GPU
#
# # Sequential NN
# ma_nn = Sequential([Dense(64,input_shape=(buff,),activation='relu'),Dense(32,activation='linear'),Dense(1)])
# ma_nn.compile(loss='mse',optimizer='rmsprop',metrics=['mae','mse'])
# history = ma_nn.fit(X_train3, y_train3, epochs=250, batch_size=32, validation_split=0.25)
# y_nn = ma_nn.predict(X_test3)
# y_nn2 = pd.Series(y_nn[:,0],index=y_test3.index)
# print("%.2f" % mse(y_nn2,y_test3))
# print('MAPE: ',"%.2f" % mape(y_nn2,y_test3))
