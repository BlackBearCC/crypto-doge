import os
from datetime import timedelta
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import backtrader as bt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class MyLSTMStrategy(bt.Strategy):
    params = (
        ('timestamp', 5),
        ('model_path', 'quant_model.h5'),
        ('initial_money', 10000),
        ('buy_amount', 1000),
        ('max_sell', 10),
        ('stop_loss', 0.03),
        ('take_profit', 0.07),
        ('trend_window', 48),  # 预测窗口调整为48小时
    )

    def __init__(self):
        self.modelnn = tf.keras.models.load_model(self.params.model_path)
        print("模型已加载:", self.params.model_path)

        self.minmax = MinMaxScaler()
        self.cash = self.params.initial_money
        self.current_inventory = 0
        self.states_buy = []
        self.states_sell = []
        self.portfolio_value = [self.cash]
        self.data_history = []
        self.trades = []  # 存储每笔交易的详情
        self.predicted_prices = []  # 用于存储预测的收盘价
        self.trend_list = []  # 用于存储每个预测窗口的趋势
        self.market_states = []  # 用于存储每个预测窗口的市场状态

        self.future_prices_all = []  # 用于存储所有预测的价格数据
        self.peaks_all = []  # 用于存储所有波峰的位置
        self.valleys_all = []  # 用于存储所有波谷的位置
        self.future_datetimes_all = []  # 用于存储每次预测的时间序列

        self.processed_trades = set()  # 存储已处理的波谷-波峰对

        self.order = None

    def predict_future(self, future_hours, last_data, modelnn, minmax, timestamp):
        future_predictions = []

        # 使用最后 timestamp 个时间步的 close_price 数据
        close_price = last_data[-timestamp:]

        # 对最近的 close_price 进行归一化处理
        close_price_scaled = minmax.transform(np.array(close_price).reshape(-1, 1)).flatten()

        for _ in range(future_hours):
            # 准备输入数据
            input_data = np.column_stack((np.zeros(timestamp),  # 假设 Polarity, Sensitivity, Tweet_vol 为 0
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

    def identify_trend(self, future_predictions):
        peaks, _ = find_peaks(future_predictions)
        valleys, _ = find_peaks(-np.array(future_predictions))

        trend_segments = []
        trends = ['sideways'] * len(future_predictions)

        # 如果没有找到波峰或波谷，假设为平稳趋势
        if len(peaks) == 0 and len(valleys) == 0:
            return trends, peaks, valleys

        # 将波峰和波谷的位置整合，并按顺序排列
        trend_points = sorted(peaks.tolist() + valleys.tolist())

        for i in range(1, len(trend_points)):
            start = trend_points[i - 1]
            end = trend_points[i]
            if future_predictions[end] > future_predictions[start]:
                trends[start:end + 1] = ['uptrend'] * (end - start + 1)
            else:
                trends[start:end + 1] = ['downtrend'] * (end - start + 1)

        return trends, peaks, valleys

    def next(self):
        # 收集当前的数据点
        self.data_history.append({
            'close': self.data.close[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'volume': self.data.volume[0]
        })
        if self.order:
            return

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

        if len(self.predicted_prices) >= self.params.trend_window:
            future_prices = self.predicted_prices[-self.params.trend_window:]
            trend_segments, peaks, valleys = self.identify_trend(future_prices)

            # 保存预测的价格和对应的时间序列
            last_datetime = self.datas[0].datetime.datetime(-1)
            future_datetimes = pd.date_range(last_datetime + timedelta(hours=1), periods=self.params.trend_window,
                                             freq='H')
            self.future_prices_all.append(future_prices)
            self.peaks_all.append(peaks)
            self.valleys_all.append(valleys)
            self.future_datetimes_all.append(future_datetimes)

            for i in range(min(len(valleys), len(peaks))):
                if peaks[i] > valleys[i]:  # 确保每个波峰在其对应的波谷之后
                    actual_valley_index = len(self.data_history) - self.params.trend_window + valleys[i]
                    actual_peak_index = len(self.data_history) - self.params.trend_window + peaks[i]

                    if (actual_valley_index, actual_peak_index) in self.processed_trades:
                        continue  # 已经处理过，跳过

                    buy_price = self.data.close[-(self.params.trend_window - valleys[i])]
                    sell_price = self.data.close[-(self.params.trend_window - peaks[i])]

                    print(
                        f"波谷时间: {self.data.datetime.datetime(-self.params.trend_window + valleys[i])}, 波谷预测价格: {future_prices[valleys[i]]}, 实际价格: {buy_price}")
                    print(
                        f"波峰时间: {self.data.datetime.datetime(-self.params.trend_window + peaks[i])}, 波峰预测价格: {future_prices[peaks[i]]}, 实际价格: {sell_price}")

                    if self.current_inventory == 0 and self.cash >= self.params.buy_amount:
                        buy_units = self.params.buy_amount / buy_price
                        self.cash -= self.params.buy_amount
                        self.current_inventory += buy_units
                        self.states_buy.append(actual_valley_index)
                        print(
                            f"执行买入操作: {buy_units:.6f} 单位，买入价格为 {buy_price:.2f}, 持仓 {self.current_inventory:.6f}")

                        # 记录买入交易
                        self.trades.append({
                            'datetime': self.data.datetime.datetime(-1),
                            'price': buy_price,
                            'size': buy_units,
                            'action': 'buy'
                        })

                    if self.current_inventory > 0 and actual_peak_index > actual_valley_index:
                        sell_units = min(self.current_inventory, self.params.max_sell)
                        self.cash += sell_units * sell_price
                        self.current_inventory -= sell_units
                        self.states_sell.append(actual_peak_index)
                        print(
                            f"执行卖出操作: {sell_units:.6f} 单位，卖出价格为 {sell_price:.2f}, 持仓 {self.current_inventory:.6f}")

                        # 记录卖出交易
                        self.trades.append({
                            'datetime': self.data.datetime.datetime(-1),
                            'price': sell_price,
                            'size': sell_units,
                            'action': 'sell'
                        })

                        # 计算并打印每笔交易的利润
                        trade_profit = (sell_price - buy_price) * sell_units
                        print(f"交易利润: {trade_profit:.2f}")

                    self.processed_trades.add((actual_valley_index, actual_peak_index))

                    # 完成一笔交易后退出循环
                    break

        # 更新投资组合的总价值
        portfolio_value = self.cash + self.current_inventory * self.data.close[0]
        self.portfolio_value.append(portfolio_value)

    def stop(self):
        # 打印最终投资组合价值
        final_value = self.portfolio_value[-1]
        print(f"最终资金: {final_value:.2f}")

        # 基于已有数据预测未来24H的价格
        future_predictions = []
        last_data = [d['close'] for d in self.data_history[-self.params.timestamp:]]
        last_datetime = self.datas[0].datetime.datetime(-1)  # 获取当前数据的最后一个时间点
        for i  in range(24):
            last_data_scaled = self.minmax.transform(np.array(last_data).reshape(-1, 1)).flatten()
            input_data = np.column_stack((np.zeros(self.params.timestamp),
                                          np.zeros(self.params.timestamp),
                                          np.zeros(self.params.timestamp),
                                          last_data_scaled))
            prediction = self.modelnn.predict(input_data[np.newaxis, :, :])
            predicted_close_price_scaled = prediction[0, -1]
            predicted_close_price = self.minmax.inverse_transform([[predicted_close_price_scaled]])[0, 0]
            future_predictions.append(predicted_close_price)

            # 计算预测时间
            predicted_time = last_datetime + timedelta(hours=i + 1)
            print(f"预测价格: {predicted_close_price:.2f}, 预测时间: {predicted_time}")

            last_data = last_data[1:] + [predicted_close_price]  # 滑动窗口更新

        # 绘制完整价格线
        plt.figure(figsize=(15, 5))
        actual_prices = [d['close'] for d in self.data_history]
        plt.plot(actual_prices, color='blue', lw=2, label='Actual Price')
        plt.plot(range(len(actual_prices), len(actual_prices) + 24), future_predictions, color='orange', lw=2,
                 label='Future Prediction (24 H)')
        plt.title('Complete Price Line with Future Predictions')
        plt.legend()
        plt.show()

        # 计算最大回撤
        portfolio_value = np.array(self.portfolio_value)
        running_max = np.maximum.accumulate(portfolio_value)
        drawdown = (running_max - portfolio_value) / running_max
        max_drawdown = np.max(drawdown)

        print(f"最大回撤: {max_drawdown * 100:.2f}%")

        # 计算胜率和总盈利
        total_trades = len(self.trades) // 2
        win_trades = sum([1 for i in range(1, len(self.trades), 2) if self.trades[i]['price'] > self.trades[i - 1]['price']])
        win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
        total_gains = final_value - self.params.initial_money
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

        # 绘制未来预测的价格趋势和波峰、波谷
        plt.figure(figsize=(15, 5))

        # 循环绘制所有收集到的预测价格和波峰、波谷
        for i in range(len(self.future_prices_all)):
            future_prices = self.future_prices_all[i]
            future_datetimes = self.future_datetimes_all[i]
            peaks = self.peaks_all[i]
            valleys = self.valleys_all[i]

            plt.plot(future_datetimes, future_prices, color='orange')


            plt.plot(future_datetimes[peaks], np.array(future_prices)[peaks], "x", color='green',
                     label='Peaks' if i == 0 else "")
            plt.plot(future_datetimes[valleys], np.array(future_prices)[valleys], "o", color='red',
                     label='Valleys' if i == 0 else "")

        plt.title('Predicted Future Prices with Identified Peaks and Valleys')
        plt.xlabel('Time')
        plt.ylabel('Price')
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

        # 打印最终资金和持仓
        print(f"最终资金: {self.cash:.2f}")
        if self.current_inventory > 0:
            print(f"最终持仓: {self.current_inventory:.6f} 单位，当前价格: {self.data.close[0]:.2f}")

        # 计算交易结果
        total_profit = 0
        for i in range(1, len(self.trades), 2):
            buy_trade = self.trades[i - 1]
            sell_trade = self.trades[i]
            if buy_trade['action'] == 'buy' and sell_trade['action'] == 'sell':
                profit = (sell_trade['price'] - buy_trade['price']) * buy_trade['size']
                total_profit += profit
                print(f"交易 {i // 2 + 1}: 买入价格 {buy_trade['price']:.2f}，卖出价格 {sell_trade['price']:.2f}，利润 {profit:.2f}")

        print(f"总利润: {total_profit:.2f}")

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
minmax = MinMaxScaler().fit(df_resampled[['close']].to_numpy().astype('float32'))

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


cerebro.run()


