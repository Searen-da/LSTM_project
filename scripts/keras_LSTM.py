"Takes in data, prepares it for analysis of forcastin time series"
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from scikeras.wrappers import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor


# total number of samples
sample_size = 1000
train_size = 700
val_size = 200
#total size of data is sample-window_size
# windwov size: use k previous samples to predict the next n sample
wd_size = 100

type = 'multi'
# single, multi
multiva = 'download'
# ts(timestamp), download
scale = False
# True, Flase
callback = False




def read_data(path):
    """read data and transforms it to NumPy array"""
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    df = pd.read_csv(path)
    small_table = df.head(sample_size)[['时间戳', '上行流量_M', '下行流量_M']]

    small_table['时间戳'] = pd.to_datetime(df['时间戳'])
    # temp = small_table['上行流量_M']
    # temp.plot()
    # small_table.plot(x='时间戳', y='上行流量_M', kind='line', title='upload_M vs Time')

    # correlation analysis
    correlation = small_table['上行流量_M'].corr(small_table['下行流量_M'])
    print(correlation)

    # normalize data
    scaler = MinMaxScaler(feature_range = (10, 15))
    if scale:
        data = small_table[['上行流量_M']]
        scaled_data = scaler.fit_transform(data)
        small_table['上行流量_M'] = scaled_data
        original_data = scaler.inverse_transform(scaled_data)

    if type == 'single':
        # for 上行流量, train/val/test set
        x1, y1 = df_to_array(small_table, wd_size)
        x1_train, y1_train, x1_val, y1_val, x1_test, y1_test = partition(x1, y1)
        run_test(x1_train, y1_train, x1_test, y1_test, x1_val, y1_val, small_table, scaler)

    elif type == 'multi':
        # for multivariate, use 上行流量,下行流量,时间 to predict 上行流量
        x1, y1 = df_to_array_multi(small_table, wd_size)
        x1_train, y1_train, x1_val, y1_val, x1_test, y1_test = partition(x1, y1)
        run_test_multi(x1_train, y1_train, x1_test, y1_test, x1_val, y1_val, small_table, scaler)

    return df

def df_to_array(small_table, window_size):
    # Convert the DataFrame to a NumPy array
    df_as_np = small_table['上行流量_M'].to_numpy()

    # Initialize empty lists for X and y
    X = []
    y = []

    # Iterate through the data with a sliding window of size 'window_size'
    for i in range(len(df_as_np) - window_size):
        # Extract the window of rows and convert them into a nested list
        window = [[a] for a in df_as_np[i:i+window_size]]
        X.append(window)

        # Extract the label for the next row after the window
        label = df_as_np[i + window_size]
        y.append(label)

    return np.array(X), np.array(y)

def df_to_array_multi(small_table, window_size):
    if multiva == 'download':
        temp_df = small_table[['上行流量_M', '下行流量_M']]
        # plt.plot(small_table['时间戳'], temp_df['上行流量_M'], label='upload_M', color='blue')
        # plt.plot(small_table['时间戳'], temp_df['下行流量_M'], label='download_M', color='orange')
        # plt.show()
    elif multiva == 'ts':
        temp = small_table['上行流量_M']
        temp_df = pd.DataFrame({'upload': temp})
        temp_df['Hours'] = (small_table['时间戳'].map(pd.Timestamp.timestamp) /3600.0 - 457488) % 24
        correlation = temp_df['upload'].corr(temp_df['Hours'])
        print(correlation)

    # Convert the DataFrame to a NumPy array
    df_as_np = temp_df.to_numpy()

    # Initialize empty lists for X and y
    X = []
    y = []

    # Iterate through the data with a sliding window of size 'window_size'
    for i in range(len(df_as_np) - window_size):
        # Extract the window of rows and convert them into a nested list
        window = [r for r in df_as_np[i:i+window_size]]
        X.append(window)

        # Extract the label for the next row after the window
        label = df_as_np[i + window_size]
        y.append(label[0])

    return np.array(X), np.array(y)

def run_test(x1_train, y1_train, x1_test, y1_test, x1_val, y1_val, small_table, scaler):
    """build a model and run it on the data"""
    # # Create the LSTM model
    # model = Sequential()
    # # model.add(LSTM(128, input_shape=(wd_size, 1)))
    # model.add(LSTM(128, return_sequences=True, input_shape=(wd_size, 1)))
    # model.add(LSTM(64))
    # # model.add(Flatten())
    # # model.add(Dense(128, 'relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(8, 'relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='linear'))
    # model.summary()
    # # Compile the model
    # # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    # model.compile(optimizer=optimizer, loss='mae',  metrics=['mean_absolute_percentage_error'])



    # grid search code
    def create_model(dropout_rate):
        # dropout_rate = 0.2
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(wd_size, 1)))
        model.add(LSTM(64))
        model.add(Dropout(dropout_rate))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error')
        return model

    # Create the model object
    grid_model = KerasRegressor(build_fn=create_model, epochs=10, batch_size=4, verbose=1)
    param_grid = {
        'dropout_rate': [0.2,0.7],
    }
    # Create a GridSearchCV object with TimeSeriesSplit cross-validation
    grid_search = GridSearchCV(estimator=grid_model, param_grid=param_grid, cv=2)

    # Fit the data to perform the grid search
    grid_search.fit(x1_train, y1_train)

    # Get the best hyperparameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    model = grid_search.best_estimator_



    if callback:
        # Define the ModelCheckpoint callback
        checkpoint_path = 'best_model_weights.h5'
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1)
        model.fit(x1_train, y1_train, validation_data=(x1_val, y1_val), epochs=20, batch_size=16, callbacks=[cp_callback])
        # predictions = model1.predict(x1_train).flatten()
        model.load_weights('best_model_weights.h5')

    if scale:
        predictions = model.predict(x1_train)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(train_size)
        y1_train = scaler.inverse_transform(y1_train.reshape(-1, 1)).reshape(train_size)
    else:
        predictions = model.predict(x1_train).reshape(train_size)
    # Print the predicted values
    # visualize(predics, small_table)
    predics = pd.DataFrame(data={'Train Predictions': predictions, 'Actual': y1_train})
    plt.plot(small_table['时间戳'][:train_size], predics['Train Predictions'], label='Train Predictions')
    plt.plot(small_table['时间戳'][:train_size], predics['Actual'], label='Actual')
    plt.legend()
    plt.show()
    print('Predictions:', predics)
    loss = mape_loss(y1_train, predictions)
    print('Test loss:', loss, "\n")

    #Model on validation data
    if scale:
        predictions1 = model.predict(x1_val)
        predictions1 = scaler.inverse_transform(predictions1.reshape(-1, 1)).reshape(val_size)
        y1_val = scaler.inverse_transform(y1_val.reshape(-1, 1)).reshape(val_size)
    else:
        predictions1 = model.predict(x1_val).reshape(val_size)
    predics1 = pd.DataFrame(data={'Train Predictions': predictions1, 'Actual': y1_val})
    # Print the predicted values
    plt.plot(small_table['时间戳'][train_size:train_size+val_size], predics1['Train Predictions'], label='Train Predictions')
    plt.plot(small_table['时间戳'][train_size:train_size+val_size], predics1['Actual'],label='Actual')
    plt.legend()
    plt.show()
    print('Predictions:', predics1)
    loss1 = mape_loss(y1_val, predictions1)
    print('Test loss for validation:', loss1)

    # To do: Model on test data

def run_test_multi(x1_train, y1_train, x1_test, y1_test, x1_val, y1_val, small_table, scaler):
    """build a model and run it on the data"""
    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(wd_size, 2)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    # model.add(Flatten())
    model.add(Dense(128, 'relu'))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()

    # Compile the model
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.002)
    # model.compile(optimizer=optimizer, loss='mae')
    model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error')
    #
    # def create_multi(dropout_rate, optimizer, learning_rate):
    #     model = Sequential()
    #     model.add(LSTM(128, return_sequences=True, input_shape=(wd_size, 2)))
    #     model.add(LSTM(64, return_sequences=True))
    #     model.add(Dropout(0.2))
    #     model.add(LSTM(64, return_sequences=True))
    #     model.add(Dropout(0.2))
    #     model.add(LSTM(32))
    #     # model.add(Flatten())
    #     model.add(Dropout(0.2))
    #     model.add(Dense(128, 'relu'))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(8, 'relu'))
    #     model.add(Dense(1, activation='linear'))
    #     model.summary()
    #
    #     # Compile the model
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #     # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    #     # model.compile(optimizer=optimizer, loss='mae')
    #     model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error')
    #     return model

    # ModelCheckpoint callback
    checkpoint_path = 'best_multi_model_weights.h5'
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1)
    model.fit(x1_train, y1_train, validation_data=(x1_val, y1_val), epochs=400, batch_size=8, callbacks=[cp_callback])
    model.load_weights('best_multi_model_weights.h5')

    # model = KerasRegressor(build_fn= create_multi, epochs=100, batch_size=4, verbose=1)
    # # define the grid search parameters
    # param_grid = {
    #     'dropout_rate': [0.2, 0.4, 0.6],
    #     'optimizer': ['adam', 'rmsprop'],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     # 'hidden_units': [64, 128, 256],
    #     # 'batch_size': [16, 32, 64]
    # }
    #
    # # # Create Grid Search
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
    # # Fit the data to perform the grid search
    # grid_search.fit(x1_train, y1_train)
    # # Get the best hyperparameters and best score
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    # model = grid_search.best_estimator_

    # # tensorboard config
    # file_name = 'kpi_multi_model'
    # tensorboard = TensorBoard(log_dir="logs/{}".format(file_name))
    # history = model.fit(x1_train, y1_train, validation_data=(x1_val, y1_val), epochs=150, batch_size=16, callbacks=[tensorboard])

    if scale:
        predictions = model.predict(x1_train)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(train_size)
        y1_train = scaler.inverse_transform(y1_train.reshape(-1, 1)).reshape(train_size)
    else:
        predictions = model.predict(x1_train).reshape(train_size, )

    predics = pd.DataFrame(data={'Train Predictions': predictions, 'Actual': y1_train})


    # Print the predicted values
    plt.plot(small_table['时间戳'][:train_size], predics['Train Predictions'], label='Train Predictions')
    plt.plot(small_table['时间戳'][:train_size], predics['Actual'], label='Actual')
    plt.legend()
    plt.show()
    plt.savefig('train.png')
    print('Predictions:', predics)

    # Evaluate the model on the testing data
    loss = mape_loss(y1_train, predictions)
    print('Test loss:', loss, "\n")

    #Model on validation data
    if scale:
        predictions1 = model.predict(x1_val)
        predictions1 = scaler.inverse_transform(predictions1.reshape(-1, 1)).reshape(val_size)
        y1_val = scaler.inverse_transform(y1_val.reshape(-1, 1)).reshape(val_size)
    else:
        predictions1 = model.predict(x1_val).reshape(val_size, )

    predics1 = pd.DataFrame(data={'Train Predictions': predictions1, 'Actual': y1_val})
    # Print the predicted values
    plt.plot(small_table['时间戳'][train_size:train_size+val_size], predics1['Train Predictions'], label='Train Predictions')
    plt.plot(small_table['时间戳'][train_size:train_size+val_size], predics1['Actual'],label='Actual')
    plt.legend()
    plt.show()
    plt.savefig('validation.png')
    print('Predictions:', predics1)
    loss1 = mape_loss(y1_val, predictions1)
    print('Test loss for validation:', loss1)

    # To do: Model on test data

def partition(x1, y1):
    print(x1.shape, y1.shape)
    # normalize data
    x1_train, y1_train = x1[:train_size], y1[:train_size]
    print(x1_train.shape, y1_train.shape)

    x1_val, y1_val = x1[train_size:train_size + val_size], y1[train_size:train_size + val_size]
    print(x1_val.shape, y1_val.shape)

    x1_test, y1_test = x1[train_size + val_size:], y1[train_size + val_size:]
    # x1_test, y1_test, scaler_test = normalize(x1_test, y1_test)
    print(x1_test.shape, y1_test.shape)
    return x1_train, y1_train, x1_val, y1_val, x1_test, y1_test

def mape_loss(y_true, y_pred):
    error = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(error) * 100
    return mape
