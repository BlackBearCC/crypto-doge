import os
from datetime import timedelta
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import backtrader as bt
import matplotlib.pyplot as plt
import backtrader.indicators as btind

class MyLSTMStrategy(bt.Strategy):
    params = (
        ('timestamp', 5),
        ('model_path', 'quant_model.h5'),
        ('initial_money', 10000),
        ('trade_amount', 200),  # 每次交易的固定金额
        ('atr_period', 14),  # ATR计算周期
        ('atr_threshold', 500),  # ATR阈值
        ('max_trade_size', 0.5),
    )

    def __init__(self):
        self.modelnn = tf.keras.models.load_model(self.params.model_path)
        print("模型已加载:", self.params.model_path)

        self.minmax = MinMaxScaler()
        self.data_history = []
        self.atr = btind.AverageTrueRange(self.data, period=self.params.atr_period)
        self.order = None  # 用于跟踪挂单

    def next(self):
        # 收集当前的数据点
        self.data_history.append({
            'close': self.data.close[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'volume': self.data.volume[0],
            'datetime': self.data.datetime.datetime(0)  # 记录当前时间
        })

        if len(self.data_history) < self.params.timestamp + self.params.atr_period:
            return

        # ATR过滤：仅在波动率高于阈值时执行交易
        if self.atr[0] < self.params.atr_threshold:
            print(f"波动率过低（ATR: {self.atr[0]:.6f}），跳过交易。")
            return

        # 预测未来1小时的方向（上涨或下跌）
        direction, prediction_time = self.predict_direction()

        # 获取账户当前资金
        current_cash = self.broker.get_cash()

        # 限制交易规模
        trade_amount = min(self.params.trade_amount, current_cash * self.params.max_trade_size)

        # 平仓操作并开新仓
        if direction > 0:  # 预测价格上涨
            if self.position.size < 0:  # 如果持有空头仓位，先平仓
                print(f"当前持有空头仓位，执行回补操作: 当前价格 {self.data.close[0]:.2f}")
                self.close()  # 平空头仓位
            if not self.position:  # 检查是否已平仓
                print(f"预测价格上涨，执行买入操作: 当前价格 {self.data.close[0]:.2f}")
                self.buy(size=trade_amount / self.data.close[0])  # 计算买入数量并执行买入
        elif direction < 0:  # 预测价格下跌
            if self.position.size > 0:  # 如果持有多头仓位，先平仓
                print(f"当前持有多头仓位，执行卖出操作: 当前价格 {self.data.close[0]:.2f}")
                self.close()  # 平多头仓位
            if not self.position:  # 检查是否已平仓
                print(f"预测价格下跌，执行开空操作: 当前价格 {self.data.close[0]:.2f}")
                self.sell(size=trade_amount / self.data.close[0])  # 计算卖出数量并执行开空单

    def predict_direction(self):
        # 使用模型预测未来1小时的方向
        close_price = [d['close'] for d in self.data_history[-self.params.timestamp:]]
        close_price_scaled = self.minmax.fit_transform(np.array(close_price).reshape(-1, 1)).flatten()
        input_data = np.column_stack((np.zeros(self.params.timestamp),  # 假设 Polarity, Sensitivity, Tweet_vol 为 0
                                      np.zeros(self.params.timestamp),
                                      np.zeros(self.params.timestamp),
                                      close_price_scaled))
        prediction = self.modelnn.predict(input_data[np.newaxis, :, :])
        predicted_close_price_scaled = prediction[0, -1]
        predicted_close_price = self.minmax.inverse_transform([[predicted_close_price_scaled]])[0, 0]

        # 预测价格方向：上涨返回1，下跌返回-1
        current_price = self.data.close[0]
        direction = 1 if predicted_close_price > current_price else -1

        prediction_time = self.data.datetime.datetime(0) + timedelta(hours=1)  # 预测时间为当前时间+1小时
        print(f"预测未来1小时的方向为: {'上涨' if direction > 0 else '下跌'}, 预测时间: {prediction_time}")
        return direction, prediction_time

    def stop(self):
        # 打印最终资金
        final_value = self.broker.getvalue()
        print(f"最终资金: {final_value:.2f}")

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

# 打印数据预览
print("数据预览:\n", df_resampled.head())
import backtrader.analyzers as btanalyzers
# 设置Cerebro并添加策略
cerebro = bt.Cerebro()
data = bt.feeds.PandasData(dataname=df_resampled)
cerebro.adddata(data)
cerebro.addstrategy(MyLSTMStrategy)
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)

# 添加分析器
cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trade_analyzer')
cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
cerebro.addanalyzer(btanalyzers.TimeReturn, _name='time_return')

# 运行策略并获取分析结果
results = cerebro.run()
strategy = results[0]

# 打印分析结果
trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()
drawdown = strategy.analyzers.drawdown.get_analysis()
sqn = strategy.analyzers.sqn.get_analysis()
time_return = strategy.analyzers.time_return.get_analysis()

print("Trade Analysis:")
print(f"总交易次数: {trade_analyzer.total.closed}")
print(f"盈利交易次数: {trade_analyzer.won.total}")
print(f"亏损交易次数: {trade_analyzer.lost.total}")
print(f"胜率: {trade_analyzer.won.total / trade_analyzer.total.closed * 100:.2f}%")
print(f"总盈利: {trade_analyzer.pnl.net.total:.2f}")
print(f"最大单笔盈利: {trade_analyzer.won.pnl.max:.2f}")
print(f"最大单笔亏损: {trade_analyzer.lost.pnl.max:.2f}")

print("\nDrawdown Analysis:")
print(f"最大回撤: {drawdown.max.drawdown:.2f}%")
print(f"最大回撤金额: {drawdown.max.moneydown:.2f}")
print(f"回撤持续时间: {drawdown.max.len} 根K线")

print("\nSQN Analysis:")
print(f"SQN (系统质量数): {sqn.sqn:.2f}")

# print("\nTime Return Analysis:")
# for period, ret in time_return.items():
#     print(f"{period.capitalize()} Return: {ret:.2f}%")

# 绘制策略表现
cerebro.plot(style='candlestick')
