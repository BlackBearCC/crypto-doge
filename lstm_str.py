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
        ('trade_amount', 1000),  # 每次交易的固定金额
        ('atr_period', 14),  # ATR计算周期
        ('atr_multiplier', 1.2),  # 动态ATR阈值的倍数
        ('max_trade_size', 0.8),
        ('stop_loss_multiplier',2),
        ('take_profit_multiplier', 3),
        ('rsi_period', 7),  # RSI 计算周期
        ('rsi_overbought', 60),  # 超买阈值
        ('rsi_oversold', 40),  # 超卖阈值
    )

    def __init__(self):
        self.modelnn = tf.keras.models.load_model(self.params.model_path)
        print("模型已加载:", self.params.model_path)

        self.minmax = MinMaxScaler()
        self.data_history = []
        # ATR相关
        self.atr = btind.AverageTrueRange(self.data, period=self.params.atr_period)
        self.atr_sma = btind.SMA(self.atr, period=self.params.atr_period)  # 计算ATR的移动平均值

        self.rsi = btind.RelativeStrengthIndex(self.data, period=self.params.rsi_period)

        self.order = None  # 用于跟踪挂单
        self.stop_order = None  # 用于存储止损订单
        self.take_profit_order = None  # 用于存储止盈订单
        # 用于存储所有交易的信息
        self.closed_trades = []

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
            # 动态调整ATR阈值
        dynamic_atr_threshold = self.atr_sma[0] * self.params.atr_multiplier

        # ATR过滤：仅在波动率高于动态阈值时执行交易
        if self.atr[0] < dynamic_atr_threshold:
            print(f"波动率过低（ATR: {self.atr[0]:.6f}, 阈值: {dynamic_atr_threshold:.6f}），跳过交易。")
            return

        # 预测未来1小时的方向（上涨或下跌）
        direction, prediction_time = self.predict_direction()

        # 获取账户当前资金
        current_cash = self.broker.get_cash()

        # 限制交易规模
        trade_amount = min(self.params.trade_amount, current_cash * self.params.max_trade_size)

        # 如果当前有持仓，检查是否需要取消已有的止盈止损挂单
        if self.position:
            if self.stop_order:
                self.cancel(self.stop_order)
            if self.take_profit_order:
                self.cancel(self.take_profit_order)

        # 信号过滤逻辑
        if direction > 0:  # 预测价格上涨
            if self.rsi[0] < self.params.rsi_oversold:  # RSI 处于超卖状态
                if self.position.size < 0:  # 如果持有空头仓位，先平仓
                    print(f"当前持有空头仓位，执行回补操作: 当前价格 {self.data.close[0]:.2f}")
                    self.close()  # 平掉空头仓位
                print(f"预测价格上涨，执行买入操作: 当前价格 {self.data.close[0]:.2f}")
                self.order = self.buy(size=trade_amount / self.data.close[0])

        elif direction < 0:  # 预测价格下跌
            if self.rsi[0] > self.params.rsi_overbought:  # RSI 处于超买状态
                if self.position.size > 0:  # 如果持有多头仓位，先平仓
                    print(f"当前持有多头仓位，执行卖出操作: 当前价格 {self.data.close[0]:.2f}")
                    self.close()  # 平多头仓位

    def set_stop_loss(self, order):
        """设置止损挂单"""
        atr_value = self.atr[0]
        stop_loss_price = order.executed.price - (atr_value * self.params.stop_loss_multiplier)
        print(f"设置止损单：{stop_loss_price:.2f}")
        self.stop_order = self.sell(size=order.executed.size, exectype=bt.Order.Stop, price=stop_loss_price)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY ORDER COMPLETED: {order.executed.size} @ {order.executed.price}')
                # 在开仓订单执行后，设置止损
                self.set_stop_loss(order)
            elif order.issell():
                print(f'SELL ORDER COMPLETED: {order.executed.size} @ {order.executed.price}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('ORDER CANCELED/MARGIN/REJECTED')

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

    def notify_trade(self, order):
        if order.isclosed:
            # 将交易信息存储在 closed_trades 列表中
            self.closed_trades.append({
                'open_date': order.open_datetime(),
                'close_date': order.close_datetime(),
                'duration': order.barlen,
                'open_price': order.price,  # 使用 trade.price 记录开仓价格
                'close_price': order.price,  # 使用 trade.price 记录平仓价格
                'size': order.size,
                'gross_profit': order.pnl,
                'net_profit': order.pnlcomm
            })


    def stop(self):
        print(f"Final Portfolio Value: {self.broker.getvalue():.2f}")

        # 统一打印所有交易的信息
        print("\nClosed Trades:")
        for i, trade in enumerate(self.closed_trades, 1):
            print(f"Trade {i}:")
            print(f" - Open Date: {trade['open_date']}")
            print(f" - Close Date: {trade['close_date']}")
            print(f" - Duration: {trade['duration']} bars")
            print(f" - Open Price: {trade['open_price']:.2f}")
            print(f" - Close Price: {trade['close_price']:.2f}")
            print(f" - Size: {trade['size']}")
            print(f" - Gross Profit: {trade['gross_profit']:.2f}")
            print(f" - Net Profit: {trade['net_profit']:.2f}")

# 加载数据并初始化策略
folder_path = 'D:\\crypto-doge\\BTCUSDT-1h'
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

# 绘制策略表现
# cerebro.plot(style='candlestick')
