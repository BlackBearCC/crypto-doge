import datetime
import math
import numpy as np
import pandas as pd
import backtrader as bt


from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import backtrader.analyzers as btanalyzers



def read_and_combine_csv(directory):
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
    return combined_df



class MultiTimeFrameRSIStrategy(bt.Strategy):
    params = (
        ('rsi1_length', 14),
        ('rsi2_length', 14),
        ('rsi3_length', 14),
        ('base_buy_threshold', 30.0),
        ('base_sell_threshold', 70.0),
        ('atr_multiplier', 15),
        ('atr_length', 14),
        ('cooldown_period', 30),
        ('initial_cash', 10000),
        ('commission', 0.1),
        ('position_size', 500),
        ('min_predicted_change', -2000),
        ('max_predicted_change', 2000),
        ('model_path', 'quant_model.h5'),
        ('timestamp', 5),  # 假设模型需要5个小时的数据
    )

    def __init__(self):
        self.rsi_5m = bt.indicators.RSI(self.data0.close, period=self.params.rsi1_length)
        self.rsi_15m = bt.indicators.RSI(self.data1.close, period=self.params.rsi2_length)
        self.rsi_30m = bt.indicators.RSI(self.data2.close, period=self.params.rsi3_length)
        self.atr = bt.indicators.ATR(self.data1, period=self.params.atr_length)

        self.modelnn = tf.keras.models.load_model(self.params.model_path)
        self.minmax = MinMaxScaler()

        self.buy_cooldown = 0
        self.sell_cooldown = 0

        # 1小时数据现在是 self.data3
        self.data_1h = self.data3

        # 用于跟踪上次预测的时间
        self.last_prediction_time = None
        self.predicted_change = 0

        self.trades = []  # 用于存储交易信息
    def next(self):
        current_time = self.data0.datetime.datetime(0)
        # print(f"当前时间: {current_time}")
        # print(f"5分钟收盘价: {self.data0.close[0]}")
        # print(f"15分钟收盘价: {self.data1.close[0]}")
        # print(f"30分钟收盘价: {self.data2.close[0]}")

        # 检查是否需要进行新的预测（每小时一次）
        if self.last_prediction_time is None or (current_time - self.last_prediction_time).total_seconds() >= 3600:
            self.predicted_change, _ = self.predict_direction()
            self.last_prediction_time = current_time

        if self.buy_cooldown > 0:
            self.buy_cooldown -= 1
        if self.sell_cooldown > 0:
            self.sell_cooldown -= 1

        current_price = self.data0.close[0]
        size = self.params.position_size / current_price
        rsi_average = (self.rsi_5m[0] + self.rsi_15m[0] + self.rsi_30m[0]) / 3
        # print(f"RSI 平均值: {rsi_average:.2f}")

        base_buy_threshold = self.params.base_buy_threshold
        base_sell_threshold = self.params.base_sell_threshold

        buy_threshold, sell_threshold = self.dynamic_threshold(self.predicted_change, base_buy_threshold,
                                                               base_sell_threshold)
        # print(f"买入阈值: {buy_threshold:.2f}")
        # print(f"卖出阈值: {sell_threshold:.2f}")

        if (self.rsi_5m[0] < buy_threshold and self.rsi_15m[0] < buy_threshold and self.rsi_30m[0] < buy_threshold and
                rsi_average < buy_threshold and self.buy_cooldown == 0):
            if self.position.size < 0:
                self.close()
            buy_order = self.buy(size=size)  # 这里保存买入订单
            self.buy_cooldown = self.params.cooldown_period

            stop_price = current_price - self.atr[0] * self.params.atr_multiplier * 0.5
            take_profit_price = current_price + self.atr[0] * self.params.atr_multiplier
            self.sell(exectype=bt.Order.Stop, price=stop_price, size=size, parent=buy_order)
            self.sell(exectype=bt.Order.Limit, price=take_profit_price, size=size, parent=buy_order)
            print(f"预测变化: {self.predicted_change},买入阈值: {buy_threshold},卖出阈值: {sell_threshold}")
            print(
                f"时间: {current_time} 买入信号触发 - 订单号={buy_order.ref}, 价格: {current_price:.2f}, 止盈: {take_profit_price:.2f}, 止损: {stop_price:.2f}, 数量: {size:.4f}, 账户资金: {self.broker.getvalue():.2f}")
        elif (self.rsi_5m[0] > sell_threshold and self.rsi_15m[0] > sell_threshold and self.rsi_30m[
            0] > sell_threshold and
              rsi_average > sell_threshold and self.sell_cooldown == 0):
            if self.position.size > 0:
                self.close()
            sell_order = self.sell(size=size)  # 这里保存卖出订单
            self.sell_cooldown = self.params.cooldown_period

            stop_price = current_price + self.atr[0] * self.params.atr_multiplier * 0.5
            take_profit_price = current_price - self.atr[0] * self.params.atr_multiplier
            self.buy(exectype=bt.Order.Stop, price=stop_price, size=size, parent=sell_order)
            self.buy(exectype=bt.Order.Limit, price=take_profit_price, size=size, parent=sell_order)
            print(f"预测变化: {self.predicted_change},买入阈值: {buy_threshold},卖出阈值: {sell_threshold}")
            print(
                f"时间: {current_time} 卖出信号触发 - 订单号={sell_order.ref}, 价格: {current_price:.2f}, 止盈: {take_profit_price:.2f}, 止损: {stop_price:.2f}, 数量: {size:.4f}, 账户资金: {self.broker.getvalue():.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受，通常这些状态不会引起实际的订单变动
            return

        # 检查订单是否完成
        # if order.status == order.Completed:
        #     if order.isbuy():
        #         print(f"买入订单完成:订单号 {order.ref} 价格={order.executed.price:.2f}, 数量={order.executed.size:.4f}, 手续费={order.executed.comm:.2f}")
        #     elif order.issell():
        #         print(f"卖出订单完成:订单号 {order.ref} 价格={order.executed.price:.2f}, 数量={order.executed.size:.4f}, 手续费={order.executed.comm:.2f}")

        # 检查是否为止损订单
        if order.exectype == bt.Order.Stop:
            print(f"止损订单触发: 订单号 {order.ref} 价格={order.executed.price:.2f}, 数量={order.executed.size:.4f}, 盈利={order.executed.pnl:.2f}")

    def predict_direction(self):
        # print("开始预测方向")
        if len(self.data3) < self.params.timestamp:
            print("1小时数据不足，无法进行预测")
            return 0, self.data0.datetime.datetime(0)

        close_prices = []
        for i in range(self.params.timestamp):
            price = self.data3.close[-i]
            if math.isnan(price):
                print(f"警告：索引 {i} 处的1小时价格数据为 NaN")
                return 0, self.data0.datetime.datetime(0)
            close_prices.append(price)

        close_prices = np.array(close_prices[::-1])  # 反转数组以保持时间顺序
        # print(f"1小时收盘价: {close_prices}")

        close_prices_scaled = self.minmax.fit_transform(close_prices.reshape(-1, 1)).flatten()
        # print(f"缩放后的1小时收盘价: {close_prices_scaled}")
        input_data = np.column_stack((np.zeros(self.params.timestamp),
                                      np.zeros(self.params.timestamp),
                                      np.zeros(self.params.timestamp),
                                      close_prices_scaled))
        # print(f"输入数据形状: {input_data.shape}")
        prediction = self.modelnn.predict(input_data[np.newaxis, :, :])
        # print(f"预测结果: {prediction}")
        predicted_close_price_scaled = prediction[0, -1]
        predicted_close_price = self.minmax.inverse_transform([[predicted_close_price_scaled]])[0, 0]
        # print(f"预测收盘价: {predicted_close_price}")

        predicted_change = predicted_close_price - close_prices[-1]
        prediction_time = self.data0.datetime.datetime(0) + datetime.timedelta(hours=1)
        # print(f"预测变化: {predicted_change}")
        return predicted_change, prediction_time

    def dynamic_threshold(self, predicted_change, base_buy_threshold, base_sell_threshold):
        normalized_predicted_change = self.normalize(predicted_change, self.params.min_predicted_change,
                                                     self.params.max_predicted_change)
        adjustment_factor = 50
        if normalized_predicted_change > 0:
            buy_threshold = base_buy_threshold + adjustment_factor * normalized_predicted_change
            sell_threshold = base_sell_threshold + adjustment_factor * normalized_predicted_change

        else:
            buy_threshold = base_buy_threshold - adjustment_factor * normalized_predicted_change
            sell_threshold = base_sell_threshold - adjustment_factor * normalized_predicted_change

        return buy_threshold, sell_threshold

    def normalize(self, predicted_change, min_value, max_value):
        normalized_value = 2 * (predicted_change - min_value) / (max_value - min_value) - 1
        return normalized_value

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        if trade.isclosed:
            self.trades.append({
                'datetime': self.data0.datetime.datetime(),
                'profit': trade.pnlcomm
            })
        # 交易关闭，打印出场信息和盈利情况
        print(
            f"交易关闭: 盈利={trade.pnl:.2f}, 净利润={trade.pnlcomm:.2f}, 进场价格={trade.price:.2f}, 出场价格={trade.price:.2f}, 数量={trade.size:.4f}")

def plot_trade_profits(trades):
    datetimes = [trade['datetime'] for trade in trades]
    profits = [trade['profit'] for trade in trades]
    colors = ['blue' if profit >= 0 else 'red' for profit in profits]

    plt.figure(figsize=(12, 6))
    plt.bar(datetimes, profits, color=colors)
    plt.title('Trade Profits')
    plt.xlabel('Trade Entry Time')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    cerebro = bt.Cerebro()

    directories = {
        '5m': 'BTCUSDT-5m',
        '15m': 'BTCUSDT-15m',
        '30m': 'BTCUSDT-30m'
    }

    data_5m = None
    for timeframe in ['5m', '15m', '30m']:
        dir_path = directories[timeframe]
        combined_df = read_and_combine_csv(dir_path)
        print(f"{timeframe} 数据点数量: {len(combined_df)}")
        todate = datetime.datetime.now()
        fromdate = todate - datetime.timedelta(days=240)
        data = bt.feeds.PandasData(dataname=combined_df, fromdate=fromdate, todate=todate)

        if timeframe == '5m':
            data.plotinfo.plot = True
            data_5m = data  # 保存5分钟数据的引用
        else:
            data.plotinfo.plot = False
            data.plotinfo.subplot = True
        cerebro.adddata(data, name=timeframe)

    # 添加重采样的1小时数据
    data_1h = cerebro.resampledata(data_5m, timeframe=bt.TimeFrame.Minutes, compression=60)
    data_1h.plotinfo.plot = False
    data_1h.plotinfo.subplot = True
    data_1h.plotinfo.plot = False
    data_1h.plotinfo.subplot = True
    cerebro.broker.setcash(10000)
    cerebro.addstrategy(MultiTimeFrameRSIStrategy)
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Minutes, compression=60,
                        annualize=True)
    cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='annual_returns')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns', fund=False)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown', fund=False)
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trade_analyzer')

    results = cerebro.run()
    strat = results[0]

    sharpe_ratio = strat.analyzers.sharpe_ratio.get_analysis()
    annual_returns = strat.analyzers.annual_returns.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()

    print(f"夏普比率: {sharpe_ratio['sharperatio']}")
    print(f"年化收益率: {annual_returns}")
    print(f"总收益率: {returns}")
    print(f"最大回撤: {drawdown.max.drawdown}")
    # 计算总收益
    final_value = cerebro.broker.getvalue()  # 回测结束时的总资金
    initial_cash = 10000  # 初始资金
    profit_amount = final_value - initial_cash  # 总盈利金额
    profit_rate = (profit_amount / initial_cash) * 100  # 盈利率

    print(f"回测结束时的总资金: {final_value:.2f}")
    print(f"总盈利金额: {profit_amount:.2f}")
    print(f"盈利率: {profit_rate:.2f}%")

    print("交易信息:")
    print(f"总交易次数: {trade_analyzer.total.total}")
    print(f"盈利交易次数: {trade_analyzer.won.total}")
    print(f"亏损交易次数: {trade_analyzer.lost.total}")
    print(f"平均盈利: {trade_analyzer.won.pnl.average:.2f}")
    print(f"平均亏损: {trade_analyzer.lost.pnl.average:.2f}")
    print(f"最大单笔盈利: {trade_analyzer.won.pnl.max:.2f}")
    print(f"最大单笔亏损: {trade_analyzer.lost.pnl.max:.2f}")
    print(f"胜率: {(trade_analyzer.won.total / trade_analyzer.total.total) * 100:.2f}%")

    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 1.0
    plt.rcParams['agg.path.chunksize'] = 10000
    cerebro.plot(style='candlestick', barup='black', bardown='white', marker='o', markersize=4, markercolor='orange')
    # 绘制交易收益柱状图
    plot_trade_profits(strat.trades)