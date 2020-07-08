from models import get_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


history_points = 50
model = get_model()
model.load_weights('./data/stocks_price.h5')
x_test_n = np.load('./data/x_test.npy')
y_test = np.load('./data/y_test.npy')
x_test = np.load('./data/x_test_real.npy')
evaluation = model.evaluate(x_test_n, y_test)
print(evaluation)
x_train = np.load('./data/x_train.npy')
y_train_real = np.array([x_train[:, 0][i + history_points] for i in range(len(x_train) - history_points)])
scale_back = preprocessing.MinMaxScaler().fit(np.expand_dims(y_train_real, -1))
y_test_predicted = model.predict(x_test_n)
y_test_predicted = scale_back.inverse_transform(y_test_predicted)
y_test_real = np.array([x_test[:, 0][i + history_points] for i in range(len(x_test) - history_points)])
real_mse = np.square(np.mean(y_test_real - y_test_predicted))
print(real_mse)

plt.gcf().set_size_inches(22, 15, forward=True)
start = 0
end = -1
real = plt.plot(y_test_real[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')
plt.legend(['Real', 'Predicted'])
plt.show()