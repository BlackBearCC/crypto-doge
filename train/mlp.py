from pathlib import Path

import joblib
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator


# 读取CSV文件并预处理数据
def read_and_prepare_data(directory):
    data_folder = Path(directory)
    all_files = data_folder.glob('*.csv')
    df_list = []

    for file_path in all_files:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)
        df_list.append(df)

    combined_df = pd.concat(df_list)
    combined_df.sort_index(inplace=True)

    # 计算技术指标
    combined_df['RSI'] = RSIIndicator(combined_df['close']).rsi()
    combined_df['ATR'] = AverageTrueRange(combined_df['high'], combined_df['low'],
                                          combined_df['close']).average_true_range()
    combined_df['SMA'] = SMAIndicator(combined_df['close'], window=100).sma_indicator()

    # 创建标签：未来1小时的价格变化
    combined_df['Price_Change'] = combined_df['close'].shift(-1) - combined_df['close']

    # 删除缺失值
    combined_df.dropna(inplace=True)

    return combined_df


data = read_and_prepare_data('../WIFUSDT-15m')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 准备特征和标签
X = data[['RSI', 'ATR', 'SMA']]
y = data['Price_Change']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

# 训练神经网络回归模型
nn_model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# 评估模型
y_pred_nn = nn_model.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Neural Network Mean Squared Error: {mse_nn}')

# 保存模型
joblib.dump(nn_model, 'mlp_model.pkl')




import matplotlib.pyplot as plt

# 散点图查看预测值和实际值的关系
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_nn, alpha=0.5)
plt.xlabel('Actual Price Change')
plt.ylabel('Predicted Price Change')
plt.title('Actual vs Predicted Price Change')
plt.show()

# 预测值分布图
plt.figure(figsize=(10, 5))
plt.hist(y_pred_nn, bins=50, alpha=0.5, label='Predicted Price Change')
plt.hist(y_test, bins=50, alpha=0.5, label='Actual Price Change')
plt.xlabel('Price Change')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Actual and Predicted Price Changes')
plt.show()

# 时间序列图查看预测值和实际值
plt.figure(figsize=(15, 5))
plt.plot(y_test.values, label='Actual Price Change')
plt.plot(y_pred_nn, label='Predicted Price Change', linestyle='--')
plt.legend()
plt.title('Time Series of Actual and Predicted Price Changes')
plt.xlabel('Sample Index')
plt.ylabel('Price Change')
plt.show()

# 保存模型
joblib.dump(nn_model, '../mlp_model.pkl')