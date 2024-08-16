import os
from datetime import timedelta

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import backtrader as bt
import matplotlib.pyplot as plt

class MyLSTMStrategy(bt.Strategy):
    params = (
        ('timestamp', 5),
        ('model_path', 'quant_model.h5'),
        ('initial_money', 10000),
        ('buy_amount', 500),
        ('max_sell', 10),
        ('stop_loss', 0.03),
        ('take_profit', 0.07),
        ('trend_window', 48),  # 预测窗口调整为7天
    )

    def __init__(self):
        self.modelnn = tf.keras.models.load_model(self.params.model_path)
        print("模型已加载:", self.params.model_path)

        self.minmax = MinMaxScaler()
        self.initial_money = self.params.initial_money
        self.current_inventory = 0
        self.states_buy = []
        self.states_sell = []
        self.portfolio_value = []
        self.data_history = []
        self.trades = []  # 存储每笔交易的详情
        self.predicted_prices = []  # 用于存储预测的收盘价
        self.trend_list = []  # 用于存储每个预测窗口的趋势
        self.market_states = []  # 用于存储每个预测窗口的市场状态

    def calculate_trend(self, prices):
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return slope

    def classify_market(self, trend, threshold=0.01):
        if trend > threshold:
            return 'uptrend'
        elif trend < -threshold:
            return 'downtrend'
        else:
            return 'sideways'

    def next(self):
        # 收集当前的数据点
        self.data_history.append({
            'close': self.data.close[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'volume': self.data.volume[0]
        })

        if len(self.data_history) < self.params.timestamp + 14:
            return

        # 计算模型所需的特征
        close_price = [d['close'] for d in self.data_history[-self.params.timestamp:]]

        # 对最近的 close_price 进行归一化处理
        close_price_scaled = self.minmax.fit_transform(np.array(close_price).reshape(-1, 1)).flatten()

        # 准备模型输入
        input_data = np.column_stack((np.zeros(self.params.timestamp),  # 假设 Polarity, Sensitivity, Tweet_vol 为 0
                                      np.zeros(self.params.timestamp),
                                      np.zeros(self.params.timestamp),
                                      close_price_scaled))

        # 预测
        prediction = self.modelnn.predict(input_data[np.newaxis, :, :])
        predicted_close_price_scaled = prediction[0, -1]
        predicted_close_price = self.minmax.inverse_transform([[predicted_close_price_scaled]])[0, 0]

        # 保存预测结果
        self.predicted_prices.append(predicted_close_price)

        # 在每个固定时间点（例如每小时）进行预测，并绘制趋势
        if len(self.predicted_prices) >= self.params.trend_window:
            future_prices = self.predicted_prices[-self.params.trend_window:]
            trend = self.calculate_trend(future_prices)
            market_state = self.classify_market(trend)
            self.trend_list.append(trend)
            self.market_states.append(market_state)

        # 打印调试信息
        print(f"模型的原始预测结果（归一化）: {predicted_close_price_scaled}")
        print(f"反归一化后的预测 close 价格: {predicted_close_price}")
        print(f"当前数据 - close: {self.data.close[0]}, high: {self.data.high[0]}, low: {self.data.low[0]}, volume: {self.data.volume[0]}")
        print(f"最近的 close_price 序列: {close_price}")
        print(f"归一化后的 close_price 序列: {close_price_scaled}")

        # 基于预测的趋势生成信号并执行交易
        if predicted_close_price > self.data.close[0]:
            if self.broker.get_cash() >= self.params.buy_amount:
                buy_units = self.params.buy_amount / self.data.close[0]
                self.buy(size=buy_units)
                self.current_inventory += buy_units
                self.states_buy.append(len(self.data_history))
                self.trades.append({
                    'type': 'buy',
                    'price': self.data.close[0],
                    'size': buy_units,
                    'datetime': self.datas[0].datetime.datetime(0)
                })
                print(f"执行买入操作: {buy_units:.6f} 单位，价格为 {self.data.close[0]:.2f}, 持仓 {self.current_inventory:.6f}")
        elif predicted_close_price < self.data.close[0] and self.current_inventory > 0:
            sell_units = min(self.current_inventory, self.params.max_sell)
            self.sell(size=sell_units)
            self.current_inventory -= sell_units
            self.states_sell.append(len(self.data_history))
            self.trades.append({
                'type': 'sell',
                    'price': self.data.close[0],
                    'size': sell_units,
                    'datetime': self.datas[0].datetime.datetime(0)
            })
            print(f"执行卖出操作: {sell_units:.6f} 单位，价格为 {self.data.close[0]:.2f}, 持仓 {self.current_inventory:.6f}")

            # 止盈止损逻辑
            if (self.data.close[0] >= close_price[-1] * (1 + self.params.take_profit)) or \
               (self.data.close[0] <= close_price[-1] * (1 - self.params.stop_loss)):
                self.sell(size=self.current_inventory)
                self.trades.append({
                    'type': 'sell',
                    'price': self.data.close[0],
                    'size': self.current_inventory,
                    'datetime': self.datas[0].datetime.datetime(0)
                })
                print(f"达到止损/止盈，全部卖出，价格为 {self.data.close[0]:.2f}, 持仓清空")
                self.current_inventory = 0

        # 更新投资组合的总价值
        self.portfolio_value.append(self.broker.get_value())


    def stop(self):
        # 打印最终投资组合价值
        final_value = self.broker.get_value()
        print(f"最终资金: {final_value:.2f}")

        # 计算最大回撤
        portfolio_value = np.array(self.portfolio_value)
        running_max = np.maximum.accumulate(portfolio_value)
        drawdown = (running_max - portfolio_value) / running_max
        max_drawdown = np.max(drawdown)

        print(f"最大回撤: {max_drawdown * 100:.2f}%")

        # 计算交易结果
        profits = []
        for i in range(1, len(self.trades), 2):  # 假设买卖成对出现
            buy_trade = self.trades[i - 1]
            sell_trade = self.trades[i]
            profit = (sell_trade['price'] - buy_trade['price']) * sell_trade['size']
            profits.append(profit)
            print(f"订单详情: 买入日期: {buy_trade['datetime']}，买入价格: {buy_trade['price']:.2f}, 卖出日期: {sell_trade['datetime']}，卖出价格: {sell_trade['price']:.2f}, 盈利: {profit:.2f}")

        win_rate = sum([1 for p in profits if p > 0]) / len(profits) * 100 if profits else 0
        total_gains = sum(profits)
        invest_return = (total_gains / self.params.initial_money) * 100

        print(f"总盈利: {total_gains:.2f}")
        print(f"收益率: {invest_return:.2f}%")
        print(f"胜率: {win_rate:.2f}%")

        # 绘制资金曲线
        plt.figure(figsize=(15, 5))
        plt.plot(portfolio_value, color='blue', lw=2, label='Portfolio Value')
        plt.title(f'Portfolio Value Curve')
        plt.legend()
        plt.show()

        # 绘制最大回撤曲线
        plt.figure(figsize=(15, 5))
        plt.plot(drawdown, color='red', lw=2, label='Max Drawdown')
        plt.title('Max Drawdown Curve')
        plt.legend()
        plt.show()

        # 确保索引不越界
        valid_buy_indices = [i for i in self.states_buy if i < len(self.data_history)]
        valid_sell_indices = [i for i in self.states_sell if i < len(self.data_history)]

        # 绘制买入卖出信号
        plt.figure(figsize=(15, 5))
        plt.plot([d['close'] for d in self.data_history], color='r', lw=2., label='Close Price')
        plt.plot(valid_buy_indices, [self.data_history[i]['close'] for i in valid_buy_indices], '^', markersize=10,
                 color='m', label='buying signal')
        plt.plot(valid_sell_indices, [self.data_history[i]['close'] for i in valid_sell_indices], 'v', markersize=10,
                 color='k', label='selling signal')
        plt.title(f'Trading Signals with Max Drawdown: {max_drawdown * 100:.2f}%')
        plt.legend()
        plt.show()

        # 绘制预测收盘价和实际收盘价的对比图
        plt.figure(figsize=(15, 5))
        actual_prices = [d['close'] for d in self.data_history[self.params.timestamp - 1:]]  # 跳过前面的无效预测
        plt.plot(actual_prices, color='blue', lw=2, label='Actual Close Price')
        plt.plot(self.predicted_prices, color='orange', lw=2, label='Predicted Close Price')
        plt.title('Predicted vs Actual Close Prices')
        plt.legend()
        plt.show()

        # 绘制预测趋势与实际价格的对比图
        plt.figure(figsize=(15, 5))
        plt.plot([d['close'] for d in self.data_history], color='blue', lw=2, label='Actual Price')

        # 添加趋势区域
        for i, state in enumerate(self.market_states):
            start_index = i + self.params.timestamp  # 正确的开始索引
            end_index = start_index + self.params.trend_window  # 趋势区域的结束索引

            if state == 'uptrend':
                plt.axvspan(start_index, end_index, color='green', alpha=0.3, label='Uptrend' if i == 0 else "")
            elif state == 'downtrend':
                plt.axvspan(start_index, end_index, color='red', alpha=0.3, label='Downtrend' if i == 0 else "")
            else:
                plt.axvspan(start_index, end_index, color='yellow', alpha=0.3, label='Sideways' if i == 0 else "")

        plt.title('Predicted Trend vs Actual Price')
        plt.legend()
        plt.show()

        def predict_future(future_days, last_data, modelnn, minmax, timestamp):
            # 初始化 future_predictions 列表，用于存储未来的预测值
            future_predictions = []

            # 使用最后 timestamp 天的 close_price 数据
            close_price = last_data[-timestamp:]

            # 对最近的 close_price 进行归一化处理
            close_price_scaled = minmax.transform(np.array(close_price).reshape(-1, 1)).flatten()

            for _ in range(future_days):
                # 准备输入数据，假设 Polarity, Sensitivity, Tweet_vol 为 0
                input_data = np.column_stack((np.zeros(timestamp),
                                              np.zeros(timestamp),
                                              np.zeros(timestamp),
                                              close_price_scaled))

                # 预测
                prediction = modelnn.predict(input_data[np.newaxis, :, :])
                predicted_close_price_scaled = prediction[0, -1]

                # 将预测结果反归一化为实际价格
                predicted_close_price = minmax.inverse_transform([[predicted_close_price_scaled]])[0, 0]

                # 保存预测结果
                future_predictions.append(predicted_close_price)

                # 更新 close_price_scaled，保持与实际情况一致
                close_price = np.append(close_price[1:], predicted_close_price)
                close_price_scaled = minmax.transform(np.array(close_price).reshape(-1, 1)).flatten()

            return future_predictions

            # 使用最后的历史数据作为输入

            # 打印数据历史长度和 datetime 索引范围

            # 使用最后的历史数据作为输入

        # 设置要预测的未来小时数
        future_hours = 36  # 预测未来的小时数

        # 使用最后的历史数据作为输入
        last_data = [d['close'] for d in self.data_history]

        # 使用新逻辑预测未来小时数的价格
        future_predictions = predict_future(future_hours, last_data, self.modelnn, self.minmax, self.params.timestamp)
        print(f"未来{future_hours}小时的预测价格: {future_predictions}")

        # 获取最后一个有效的时间点
        last_datetime = self.datas[0].datetime.datetime(-1)
        print(f"最后一个时间点: {last_datetime}")

        # 获取未来的时间轴（小时级别）
        future_datetimes = pd.date_range(last_datetime + timedelta(hours=1), periods=future_hours, freq='H')
        print(f"未来的时间范围: {future_datetimes}")

        # 仅绘制预测的部分
        plt.figure(figsize=(15, 7))

        # 绘制未来预测价格
        plt.plot(future_datetimes, future_predictions, label='Predicted Future Close Price', color='orange')

        # 设置图表标题和标签
        plt.title('Predicted Future Prices for Next 48 Hours')
        plt.xlabel('Time')
        plt.ylabel('Price')
        # plt.legend()
        plt.show()

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
cerebro.addstrategy(MyLSTMStrategy)
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)

# 运行回测
print('初始资金: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('结束资金: %.2f' % cerebro.broker.getvalue())
