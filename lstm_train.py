import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

df = pd.read_csv('sentiment-bitcoin.csv')
df = df.rename(columns = {'Unnamed: 0': 'timestamp'})
df.head()


df['Polarity'].describe()

df['Sensitivity'].describe()

df['Tweet_vol'].describe()
df['Close_Price'].describe()
def detect(signal, treshold = 2.0):
    detected = []
    for i in range(len(signal)):
        if np.abs(signal[i]) > treshold:
            detected.append(i)
    return detected
#%%
signal = np.copy(df['Close_Price'].values)
std_signal = (signal - np.mean(signal)) / np.std(signal)
s = pd.Series(std_signal)
s.describe(percentiles = [0.25, 0.5, 0.75, 0.95])
#%%
outliers = detect(std_signal, 1.3)
#%%
plt.figure(figsize = (15, 7))
plt.plot(np.arange(len(signal)), signal)
plt.plot(
    np.arange(len(signal)),
    signal,
    'X',
    label = 'outliers',
    markevery = outliers,
    c = 'r',
)
plt.xticks(
    np.arange(len(signal))[::15], df['timestamp'][::15], rotation = 'vertical'
)
plt.show()
#%%
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler().fit(df[['Polarity', 'Sensitivity', 'Close_Price']])
scaled = minmax.transform(df[['Polarity', 'Sensitivity', 'Close_Price']])
#%%
plt.figure(figsize = (15, 7))
plt.plot(np.arange(len(signal)), scaled[:, 0], label = 'Scaled polarity')
plt.plot(np.arange(len(signal)), scaled[:, 1], label = 'Scaled sensitivity')
plt.plot(np.arange(len(signal)), scaled[:, 2], label = 'Scaled closed price')
plt.plot(
    np.arange(len(signal)),
    scaled[:, 0],
    'X',
    label = 'outliers polarity based on close',
    markevery = outliers,
    c = 'r',
)
plt.plot(
    np.arange(len(signal)),
    scaled[:, 1],
    'o',
    label = 'outliers polarity based on close',
    markevery = outliers,
    c = 'r',
)
plt.xticks(
    np.arange(len(signal))[::15], df['timestamp'][::15], rotation = 'vertical'
)
plt.legend()
plt.show()


def df_shift(df, lag=0, start=1, skip=1, rejected_columns=[]):
    df = df.copy()
    if not lag:
        return df
    cols = {}
    for i in range(start, lag + 1, skip):
        for x in list(df.columns):
            if x not in rejected_columns:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i += 1
        df = pd.concat([df, dfn], axis=1)
    return df

# 删除非数值列（如 'timestamp'）以便计算相关性
df_numeric = df.drop(columns=['timestamp'])
colormap = plt.cm.RdBu
plt.figure(figsize=(15, 7))
plt.title('pearson correlation', y=1.05, size=16)

mask = np.zeros_like(df_numeric.corr())
mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    df_numeric.corr(),
    mask=mask,
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor='white',
    annot=True,
)
plt.show()
#%%
df_new = df_shift(df, lag = 42, start = 7, skip = 7)
df_new.shape
#%%
colormap = plt.cm.RdBu
plt.figure(figsize = (30, 20))
ax = plt.subplot(111)
plt.title('42 hours correlation', y = 1.05, size = 16)
selected_column = [
    col
    for col in list(df_new)
    if any([k in col for k in ['Polarity', 'Sensitivity', 'Close']])
]

sns.heatmap(
    df_new[selected_column].corr(),
    ax = ax,
    linewidths = 0.1,
    vmax = 1.0,
    square = True,
    cmap = colormap,
    linecolor = 'white',
    annot = True,
)
plt.show()

def moving_average(signal, period):
    buffer = [np.nan] * period
    for i in range(period, len(signal)):
        buffer.append(signal[i - period : i].mean())
    return buffer
#%%
signal = np.copy(df['Close_Price'].values)
ma_7 = moving_average(signal, 7)
ma_14 = moving_average(signal, 14)
ma_30 = moving_average(signal, 30)
#%%
plt.figure(figsize = (15, 7))
plt.plot(np.arange(len(signal)), signal, label = 'real signal')
plt.plot(np.arange(len(signal)), ma_7, label = 'ma 7')
plt.plot(np.arange(len(signal)), ma_14, label = 'ma 14')
plt.plot(np.arange(len(signal)), ma_30, label = 'ma 30')
plt.legend()
plt.show()

#%% md
## 开始深度学习 LSTM
#%%
import random

num_layers = 1
learning_rate = 0.005
size_layer = 128
timestamp = 5
epoch = 500
dropout_rate = 0.6

#%%
dates = pd.to_datetime(df.iloc[:, 0]).tolist()
#%%
class Model:
    def __init__(self, learning_rate, num_layers, input_dim, size_layer, dropout_rate=0.6):
        self.model = tf.keras.Sequential()

        for _ in range(num_layers):
            self.model.add(tf.keras.layers.LSTM(size_layer, return_sequences=True if _ < num_layers - 1 else False))
            self.model.add(tf.keras.layers.Dropout(dropout_rate))

        self.model.add(tf.keras.layers.Dense(input_dim))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='mean_squared_error'
        )

    def train(self, X, Y, epochs, batch_size):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

#### 数据预处理
# 正则化数据
minmax = MinMaxScaler().fit(df[['Polarity', 'Sensitivity', 'Tweet_vol', 'Close_Price']].astype('float32'))
df_scaled = minmax.transform(df[['Polarity', 'Sensitivity', 'Tweet_vol', 'Close_Price']].astype('float32'))
df_scaled = pd.DataFrame(df_scaled)

df_scaled.head()

#%%
# 转换数据以适应 LSTM 模型
X_train = []
Y_train = []

for i in range(timestamp, len(df_scaled)):
    X_train.append(df_scaled.iloc[i-timestamp:i].values)
    Y_train.append(df_scaled.iloc[i].values)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# 创建并训练模型
modelnn = Model(learning_rate, num_layers, df_scaled.shape[1], size_layer, dropout_rate)
modelnn.train(X_train, Y_train, epochs=epoch, batch_size=32)

predictions = modelnn.predict(X_train)

#%% md
#### 预测未来的时间序列数据
#%%
def predict_future(future_count, df, dates, modelnn, indices={}):
    date_ori = dates[:]
    cp_df = df.copy()
    output_predict = np.zeros((cp_df.shape[0] + future_count, cp_df.shape[1]))
    output_predict[0] = cp_df.iloc[0]
    upper_b = (cp_df.shape[0] // timestamp) * timestamp

    for k in range(0, (df.shape[0] // timestamp) * timestamp, timestamp):
        out_logits = modelnn.predict(np.expand_dims(cp_df.iloc[k : k + timestamp].values, axis=0))
        output_predict[k + 1 : k + timestamp + 1] = out_logits

    out_logits = modelnn.predict(np.expand_dims(cp_df.iloc[upper_b:].values, axis=0))
    output_predict[upper_b + 1 : cp_df.shape[0] + 1] = out_logits

    cp_df.loc[cp_df.shape[0]] = out_logits[-1]
    date_ori.append(date_ori[-1] + timedelta(hours=1))

    if indices:
        for key, item in indices.items():
            cp_df.iloc[-1, key] = item

    for i in range(future_count - 1):
        out_logits = modelnn.predict(np.expand_dims(cp_df.iloc[-timestamp:].values, axis=0))
        output_predict[cp_df.shape[0]] = out_logits[-1]
        cp_df.loc[cp_df.shape[0]] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(hours=1))

        if indices:
            for key, item in indices.items():
                cp_df.iloc[-1, key] = item

    return {'date_ori': date_ori, 'df': cp_df.values}

#%% md
#### 定义平滑函数
#%%
def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

#%% md
#### 预测并绘制结果
#%%
predict_30 = predict_future(30, df_scaled, dates, modelnn)
predict_30['df'] = minmax.inverse_transform(predict_30['df'])

#%%
plt.figure(figsize=(15, 7))
plt.plot(
    np.arange(len(predict_30['date_ori'])),
    anchor(predict_30['df'][:, -1], 0.5),
    label='predict signal',
)
# 使用原始的 df 数据框来获取 'Close_Price' 列
plt.plot(np.arange(len(df)), df['Close_Price'], label='real signal')
plt.legend()
plt.show()
# 保存整个模型（包括架构和权重）
modelnn.model.save('quant_model.h5')


##使用抱抱脸数据集测试
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import timedelta

# 读取Parquet格式的数据
df_test  = pd.read_parquet('BTC_USDT_ohlcv_data.parquet')

# 显示数据的前几行以确保数据加载正确
print(df_test .head())

#%%
# 将字符串格式的时间戳转换为datetime格式
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])

# 设置时间戳为索引
df_test.set_index('timestamp', inplace=True)

# 补充训练时使用的特征
df_test['Polarity'] = 0.0  # 新数据集中可能没有Polarity，所以用0填充
df_test['Sensitivity'] = 0.0  # 用0填充
df_test['Tweet_vol'] = 0.0  # 用0填充

# 根据训练数据中的范围生成随机数
# n = len(df_test)
# df_test['Polarity'] = np.random.uniform(0.05, 0.14, n)
# df_test['Sensitivity'] = np.random.uniform(0.17, 0.27, n)
# df_test['Tweet_vol'] = np.random.uniform(3000, 10500, n)

# 如果Close_Price没有，可以用close代替
df_test['Close_Price'] = df_test['close']  # 使用现有的close列

# 重新排列列顺序以匹配训练时的顺序
df_test = df_test[['Polarity', 'Sensitivity', 'Tweet_vol', 'Close_Price']]

# 归一化数据
minmax = MinMaxScaler().fit(df_test.astype('float32'))
df_test_scaled = minmax.transform(df_test.astype('float32'))

df_test_scaled = pd.DataFrame(df_test_scaled, columns=df_test.columns, index=df_test.index)

# 设置时间戳长度，确保与训练时一致
timestamp = 5

# 准备测试数据
X_test = []

for i in range(timestamp, len(df_test_scaled)):
    X_test.append(df_test_scaled.iloc[i-timestamp:i].values)

X_test = np.array(X_test)

# 检查X_test的形状
print("X_test shape:", X_test.shape)

#%%
# 使用训练好的模型进行预测
predictions = modelnn.predict(X_test)

# 反归一化预测结果
predictions = minmax.inverse_transform(predictions)

# 提取预测的 Close_Price 值
predicted_close = predictions[:, -1]  # 'Close_Price' 是最后一列

# 可视化结果与实际 close 值进行对比
plt.figure(figsize=(15, 7))
plt.plot(df_test.index[timestamp:], df_test['Close_Price'][timestamp:], label='Actual Close Price')
plt.plot(df_test.index[timestamp:], predicted_close, label='Predicted Close Price')
plt.legend()
plt.show()