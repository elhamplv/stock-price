# import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
import pandas as pd
from utils import get_data, slicing_50
from tensorflow.keras.utils import plot_model
from models import get_model
import numpy as np

# get data from alpha-vantage
data = get_data('*************', 'MSFT')

# data to dataframe and then numpy
df = pd.DataFrame(data)
data_np = df.to_numpy()

# split data to train, evaluation and test
n1 = int(data_np.shape[0] * 0.8)
n2 = int((data_np.shape[0] - n1)/2)
x_train = data_np[:n1]
x_val = data_np[n1: n1 + n2]
x_test = data_np[n1 + n2:]
np.save('./data/x_test_real.npy', x_test)
# standardize data to (0, 1) range
minmax_scale = preprocessing.MinMaxScaler().fit(x_train)
x_train_n = minmax_scale.transform(x_train)
x_val_n = minmax_scale.transform(x_val)
x_test_n = minmax_scale.transform(x_test)

# slice data
history_points = 50
x_train_n, y_train = slicing_50(x_train_n, history_points)
x_val_n, y_val = slicing_50(x_val_n, history_points)
x_test_n, y_test = slicing_50(x_test_n, history_points)
np.save('./data/x_train.npy', x_train_n)

# save data for later
np.save('./data/y_train.npy', y_train)
np.save('./data/x_test.npy', x_test_n)
np.save('./data/y_test.npy', y_test)

# call model
model = get_model()
# compile model
# tf.random.set_seed(4)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')

# visualize model
plot_model(model, './model-diagram.png')

# train model and save weights
mcp_save = ModelCheckpoint('./stocks_price.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(x=x_train_n, y=y_train, batch_size=32, epochs=50, shuffle=True,
                    validation_data=(x_val_n, y_val), callbacks=[mcp_save])