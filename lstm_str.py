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
        ('buy_amount', 1000),
        ('max_sell', 10),
        ('stop_loss', 0.03),
        ('take_profit', 0.07),
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

        self.order = None

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

        # 预测未来1小时的收盘价
        future_price = self.predict_future_price()

        # 根据预测的价格决定买卖
        if future_price > self.data.close[0]:
            print(f"预测价格上涨，执行买入操作: 预测价格 {future_price:.2f}, 当前价格 {self.data.close[0]:.2f}")
            self.handle_buy_signal(future_price)
        elif future_price < self.data.close[0]:
            print(f"预测价格下跌，执行卖出操作: 预测价格 {future_price:.2f}, 当前价格 {self.data.close[0]:.2f}")
            self.handle_sell_signal(future_price)

        # 更新投资组合的总价值
        portfolio_value = self.cash + self.current_inventory * self.data.close[0]
        self.portfolio_value.append(portfolio_value)

    def predict_future_price(self):
        # 使用模型预测未来1小时的价格
        close_price = [d['close'] for d in self.data_history[-self.params.timestamp:]]
        close_price_scaled = self.minmax.fit_transform(np.array(close_price).reshape(-1, 1)).flatten()
        input_data = np.column_stack((np.zeros(self.params.timestamp),  # 假设 Polarity, Sensitivity, Tweet_vol 为 0
                                      np.zeros(self.params.timestamp),
                                      np.zeros(self.params.timestamp),
                                      close_price_scaled))
        prediction = self.modelnn.predict(input_data[np.newaxis, :, :])
        predicted_close_price_scaled = prediction[0, -1]
        predicted_close_price = self.minmax.inverse_transform([[predicted_close_price_scaled]])[0, 0]
        print(f"预测未来1小时的价格为: {predicted_close_price:.2f}")
        return predicted_close_price

    def handle_buy_signal(self, predicted_price):
        # if self.current_inventory > 0:
        #     return  # 已经持仓，不重复买入

        buy_units = self.params.buy_amount / self.data.close[0]  # 计算买入单位
        self.cash -= self.params.buy_amount
        self.current_inventory += buy_units
        self.states_buy.append(len(self.data_history))
        print(f"买入操作: {buy_units:.6f} 单位，买入价格 {self.data.close[0]:.2f}, 持仓 {self.current_inventory:.6f}")

        # 记录买入交易
        self.trades.append({
            'datetime': self.data.datetime.datetime(-1),
            'price': self.data.close[0],
            'size': buy_units,
            'action': 'buy'
        })

    def handle_sell_signal(self, predicted_price):
        if self.current_inventory <= 0:
            return  # 没有持仓，不进行卖出

        sell_units = self.current_inventory  # 卖出全部持仓
        self.cash += sell_units * self.data.close[0]
        self.current_inventory -= sell_units
        self.states_sell.append(len(self.data_history))
        print(f"卖出操作: {sell_units:.6f} 单位，卖出价格 {self.data.close[0]:.2f}, 持仓 {self.current_inventory:.6f}")

        # 记录卖出交易
        self.trades.append({
            'datetime': self.data.datetime.datetime(-1),
            'price': self.data.close[0],
            'size': sell_units,
            'action': 'sell'
        })

    def stop(self):
        # 打印最终投资组合价值
        final_value = self.portfolio_value[-1]
        print(f"最终资金: {final_value:.2f}")

        # 绘制完整价格线
        plt.figure(figsize=(15, 5))
        actual_prices = [d['close'] for d in self.data_history]
        plt.plot(actual_prices, color='blue', lw=2, label='Actual Price')
        plt.title('Complete Price Line')
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
