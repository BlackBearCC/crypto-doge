import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import backtrader as bt


# 自定义策略类
class MyStrategy(bt.Strategy):
    params = (
        ('timestamp', 5),
        ('model_path', 'quant_model.h5'),
    )

    def __init__(self):
        # 加载模型
        self.modelnn = tf.keras.models.load_model(self.params.model_path)
        print("模型已加载:", self.params.model_path)

        # 创建历史数据缓存
        self.data_history = []

    def next(self):
        # 收集当前的数据点
        self.data_history.append({
            'close': self.data.close[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'volume': self.data.volume[0]
        })

        # 打印当前数据点
        print(
            f"当前数据 - close: {self.data.close[0]}, high: {self.data.high[0]}, low: {self.data.low[0]}, volume: {self.data.volume[0]}")

        # 如果历史数据不足以进行计算，直接返回
        if len(self.data_history) < self.params.timestamp:
            return

        # 计算模型所需的特征
        polarity = 0.0  # 在此处计算或获取你的 Polarity 特征
        sensitivity = 0.0  # 在此处计算或获取你的 Sensitivity 特征
        tweet_vol = 0.0  # 在此处计算或获取你的 Tweet_vol 特征
        close_price = [d['close'] for d in self.data_history[-self.params.timestamp:]]

        # 打印最近的 close_price 序列
        print("最近的 close_price 序列:", close_price)

        # 仅对 `close_price` 列进行归一化处理
        close_price_scaled = minmax.transform(np.array(close_price).reshape(-1, 1)).flatten()

        # 打印归一化后的 close_price 序列
        print("归一化后的 close_price 序列:", close_price_scaled)

        # 将特征组合在一起
        input_data = np.column_stack((np.full(self.params.timestamp, polarity),
                                      np.full(self.params.timestamp, sensitivity),
                                      np.full(self.params.timestamp, tweet_vol),
                                      close_price_scaled))

        # 预测
        prediction = self.modelnn.predict(input_data[np.newaxis, :, :])
        predicted_close_price_scaled = prediction[0, -1]

        # 打印模型的原始预测结果
        print("模型的原始预测结果（归一化）:", predicted_close_price_scaled)

        # 反归一化预测的 close 价格
        predicted_close_price = minmax.inverse_transform([[predicted_close_price_scaled]])[0, 0]

        # 打印反归一化后的预测 close 价格
        print("反归一化后的预测 close 价格:", predicted_close_price)

        # 简单的交易逻辑示例
        if predicted_close_price > close_price[-1]:
            if not self.position:
                self.buy()
                print("执行买入操作")
        elif self.position:
            self.sell()
            print("执行卖出操作")

        # 保持历史数据大小不超过 timestamp
        if len(self.data_history) > self.params.timestamp:
            self.data_history.pop(0)

# 加载数据并初始化策略
folder_path = 'D:\\crypto-doge\\BTCUSDT-1h-2024-08-01-12'
df_list = []

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(folder_path, filename), header=0)
        df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                      'taker_buy_quote_volume', 'ignore']
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df_list.append(df)

df_resampled = pd.concat(df_list)

# 归一化只对 close 进行
# 使用 numpy 数组拟合 MinMaxScaler 而不是带有列名的 DataFrame
minmax = MinMaxScaler().fit(df_resampled[['close']].to_numpy().astype('float32'))


# 打印数据预览
print("数据预览:\n", df_resampled.head())

# 设置Cerebro并添加策略
cerebro = bt.Cerebro()
data = bt.feeds.PandasData(dataname=df_resampled)
cerebro.adddata(data)
cerebro.addstrategy(MyStrategy)
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)

# 运行回测
print('初始资金: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('结束资金: %.2f' % cerebro.broker.getvalue())
cerebro.plot()