import asyncio
import datetime
import json
from pathlib import Path

import ccxt.pro
import joblib
import numpy as np
import pytz
from backtrader import Cerebro, TimeFrame
from backtrader.feeds import PandasData

from ta.momentum import RSIIndicator
import time
import pandas as pd
from matplotlib import pyplot as plt

# async def main():
#     exchange = ccxt.pro.okx()  # Create an instance of the OKX exchange
#     symbol = 'BTC/USDT'
#     timeframe = '1d'
#     limit = 100  # Maximum number of data points
#
#     try:
#         while True:
#             ohlcv = await exchange.watch_ohlcv(symbol, timeframe, limit)
#             print(ohlcv)
#             # 处理数据或继续获取
#     finally:
#         await exchange.close()  # 确保在结束时释放资源
#
# # Run the asynchronous event loop
# if __name__ == '__main__':
#     asyncio.run(main())

import backtrader as bt


# # 初始化CCXT的OKX接口
# exchange = ccxt.okx({
#     'enableRateLimit': True,
#     'countries': ['CN'],
# })
def fetch_data(symbol, timeframe, limit):
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
    df.set_index('timestamp', inplace=True)
    return df



# 读取CSV文件并预处理数据
def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('timestamp', inplace=True)
    return df

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

    # Combine all data frames into a single data frame
    combined_df = pd.concat(df_list)
    combined_df.sort_index(inplace=True)  # Make sure the index is sorted
    return combined_df



# 归一化函数
def normalize(predicted_change, min_value, max_value):
    normalized_value = 2 * (predicted_change - min_value) / (max_value - min_value) - 1
    return normalized_value

# 动态阈值函数
def dynamic_threshold(predicted_change, base_buy_threshold, base_sell_threshold, min_value, max_value):
    normalized_predicted_change = normalize(predicted_change, min_value, max_value)
    adjustment_factor = 50  # 调整因子，可以根据需要调整

    buy_threshold = base_buy_threshold + adjustment_factor * normalized_predicted_change
    sell_threshold = base_sell_threshold + adjustment_factor * normalized_predicted_change

    return buy_threshold, sell_threshold


class MultiTimeFrameRSIStrategy(bt.Strategy):
    params = (
        ('rsi1_length', 14),
        ('rsi2_length', 14),
        ('rsi3_length', 14),
        ('base_buy_threshold', 30.0),
        ('base_sell_threshold', 70.0),
        ('atr_multiplier', 16),
        ('atr_length', 14),
        ('cooldown_period', 15),
        ('initial_cash', 10000),
        ('commission', 0.1),
        ('position_size', 500),
        ('min_predicted_change', -2000),
        ('max_predicted_change', 2000),
    )

    def __init__(self):
        # 初始化技术指标
        self.rsi_5m = bt.indicators.RSI(self.datas[0].close, period=self.params.rsi1_length)
        self.rsi_15m = bt.indicators.RSI(self.datas[1].close, period=self.params.rsi2_length)
        self.rsi_30m = bt.indicators.RSI(self.datas[2].close, period=self.params.rsi3_length)
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_length)
        self.long_term_ma = bt.indicators.SMA(self.datas[0].close, period=100)
        self.bollinger = bt.indicators.BollingerBands(self.datas[0].close, period=20)
        self.ema = bt.indicators.EMA(self.datas[0].close, period=20)
        self.macd = bt.indicators.MACDHisto(self.datas[0].close)
        self.stochastic = bt.indicators.Stochastic(self.datas[0], period=14)

        self.model = joblib.load('meta_model.pkl')

        self.buy_cooldown = 0
        self.sell_cooldown = 0

        self.trade_info = {
            '总交易数': 0,
            '胜率': 0,
            '净利润': 0,
            '最大回撤': 0,
        }

        self.predicted_price_line = self.datas[0].close * 0  # 初始化一个与close数据长度相同的0数组

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        dt = pd.Timestamp(dt).tz_localize('UTC').tz_convert('Asia/Shanghai')
        formatted_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
        print('%s, %s' % (formatted_dt, txt))

    def next(self):
        if self.buy_cooldown > 0:
            self.buy_cooldown -= 1
        if self.sell_cooldown > 0:
            self.sell_cooldown -= 1

        current_price = self.datas[0].close[0]
        size = self.params.position_size / current_price
        rsi_average = (self.rsi_5m[0] + self.rsi_15m[0] + self.rsi_30m[0]) / 3

        atr_value = self.atr[0]
        sma_value = self.long_term_ma[0]

        bollinger_high = self.bollinger.lines.top[0]
        bollinger_low = self.bollinger.lines.bot[0]
        ema_value = self.ema[0]
        macd_value = self.macd.macd[0]
        stochastic_value = self.stochastic.percK[0]

        bollinger_high = self.bollinger.lines.top[0]
        bollinger_low = self.bollinger.lines.bot[0]
        ema_value = self.ema[0]
        macd_value = self.macd.macd[0]
        stochastic_value = self.stochastic.percK[0]

        # 计算滞后特征
        lag_close = self.datas[0].close[-1] if len(self.datas[0].close) > 1 else current_price
        lag_rsi = rsi_average
        lag_atr = atr_value
        lag_macd = macd_value
        lag_bb_high = bollinger_high
        lag_bb_low = bollinger_low
        lag_sma_3 = sum([self.datas[0].close[-i] for i in range(1, 4)]) / 3 if len(
            self.datas[0].close) > 3 else sma_value
        lag_sma_7 = sum([self.datas[0].close[-i] for i in range(1, 8)]) / 7 if len(
            self.datas[0].close) > 7 else sma_value

        # 时间特征
        minute = self.datas[0].datetime.datetime(0).minute
        hour = self.datas[0].datetime.datetime(0).hour
        day_of_week = self.datas[0].datetime.datetime(0).weekday()

        #22特征版本
        # # Ensure all features are floats and construct input data
        # input_data = np.array([[
        #     float(rsi_average), float(atr_value), float(macd_value), float(bollinger_high), float(bollinger_low),
        #     float(self.ema[0]), float(self.long_term_ma[0]),
        #     float(lag_close), float(lag_rsi), float(lag_atr), float(lag_macd), float(lag_bb_high), float(lag_bb_low),
        #     float(lag_sma_3), float(lag_sma_7),
        #     float(minute), float(hour), float(day_of_week),
        #     float(self.datas[0].close[-2] if len(self.datas[0].close) > 2 else current_price),
        #     float(self.datas[0].close[-3] if len(self.datas[0].close) > 3 else current_price),
        #     float(self.datas[0].close[-4] if len(self.datas[0].close) > 4 else current_price),
        #     float(self.datas[0].close[-5] if len(self.datas[0].close) > 5 else current_price)
        # ]], dtype=float)

        ##11特征版本
        input_data = np.array([[
            float(rsi_average), float(atr_value), float(macd_value), float(bollinger_high), float(bollinger_low),
            float(self.ema[0]), float(self.long_term_ma[0]),
            float(lag_close), float(lag_rsi), float(lag_atr), float(lag_macd)
        ]], dtype=float)

        print(input_data)

        predicted_change = self.model.predict(input_data)[0]

        # 计算预测方向
        predicted_direction = 'UP' if predicted_change > 0 else 'DOWN'

        base_buy_threshold = self.params.base_buy_threshold
        base_sell_threshold = self.params.base_sell_threshold

        buy_threshold, sell_threshold = dynamic_threshold(predicted_change, base_buy_threshold, base_sell_threshold,
                                                          self.params.min_predicted_change,
                                                          self.params.max_predicted_change)

        predicted_price = current_price + predicted_change
        self.predicted_price_line[0] = predicted_price
        self.log(
            f'CURRENT_PRICE: {current_price:.2f}, PREDICTED_PRICE: {predicted_price:.2f}, PREDICTED_CHANGE: {predicted_change:.2f}, PREDICTED_DIRECTION: {predicted_direction}')

        if (self.rsi_5m[0] < buy_threshold and self.rsi_15m[0] < buy_threshold and self.rsi_30m[0] < buy_threshold and
                rsi_average < buy_threshold and self.buy_cooldown == 0):
            if self.position.size < 0:
                self.close()
            self.buy(size=size)
            self.buy_cooldown = self.params.cooldown_period
            stop_price = current_price - atr_value * self.params.atr_multiplier
            take_profit_price = current_price + atr_value * self.params.atr_multiplier
            self.sell(exectype=bt.Order.Stop, price=stop_price)
            self.sell(exectype=bt.Order.Limit, price=take_profit_price)
            # self.log(
            #     f'BUY , {current_price:.2f}, SIZE: {size:.2f}, STOP: {stop_price:.2f}, LIMIT: {take_profit_price:.2f},PREDICTED_PRICE: {predicted_price:.2f},BUY_THRESHOLD: {buy_threshold:.2f},SELL_THRESHOLD: {sell_threshold:.2f}')

        elif (self.rsi_5m[0] > sell_threshold and self.rsi_15m[0] > sell_threshold and self.rsi_30m[
            0] > sell_threshold and
              rsi_average > sell_threshold and self.sell_cooldown == 0):
            if self.position.size > 0:
                self.close()
            self.sell(size=size)
            self.sell_cooldown = self.params.cooldown_period
            stop_price = current_price + atr_value * self.params.atr_multiplier
            take_profit_price = current_price - atr_value * self.params.atr_multiplier
            self.buy(exectype=bt.Order.Stop, price=stop_price)
            self.buy(exectype=bt.Order.Limit, price=take_profit_price)
            # self.log(
            #     f'SELL , {current_price:.2f}, SIZE: {size:.2f}, STOP: {stop_price:.2f}, LIMIT: {take_profit_price:.2f},PREDICTED_PRICE: {predicted_price:.2f},BUY_THRESHOLD: {buy_threshold:.2f},SELL_THRESHOLD: {sell_threshold:.2f}')

    def stop(self):
        self.trade_info['总交易数'] = self.analyzers.trade_analyzer.get_analysis().total.closed
        self.trade_info['胜率'] = self.analyzers.trade_analyzer.get_analysis().won.total / self.trade_info['总交易数']
        self.trade_info['净利润'] = self.broker.getvalue() - self.params.initial_cash
        self.trade_info['最大回撤'] = self.analyzers.drawdown.get_analysis().max.moneydown
        print(f"策略绩效指标:\n{self.trade_info}")

    def get_analysis(self):
        return self.predicted_price_line

import backtrader.analyzers as btanalyzers
if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # 指定存放CSV文件的目录
    # directories = {
    #     '5m': 'WIFUSDT-5m',
    #     '15m': 'WIFUSDT-15m',
    #     '30m': 'WIFUSDT-30m'
    # }
    directories = {
        '5m': 'BTCUSDT-5m',
        '15m': 'BTCUSDT-15m',
        '30m': 'BTCUSDT-30m'
    }


    for timeframe, dir_path in directories.items():
        combined_df = read_and_combine_csv(dir_path)
        data = bt.feeds.PandasData(dataname=combined_df)
        # 设置时间范围
        todate = datetime.datetime.now()
        fromdate = todate - datetime.timedelta(days=60)

        data = bt.feeds.PandasData(dataname=combined_df, fromdate=fromdate, todate=todate)
        # data = bt.feeds.PandasData(dataname=combined_df)

        if timeframe == '5m':
            data.plotinfo.plot = True  # Only the 5-minute data is plotted on the main plot
        else:
            data.plotinfo.plot = False  # Other timeframes are either not plotted or on subplots
            data.plotinfo.subplot = True
        cerebro.adddata(data)

    cerebro.broker.setcash(10000)  # 初始资金设为5000

    # 添加数据和策略
    cerebro.addstrategy(MultiTimeFrameRSIStrategy)
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_ratio', timeframe=TimeFrame.Minutes, compression=60,
                        annualize=True)
    cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='annual_returns')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns', fund=False)
    cerebro.addanalyzer(btanalyzers.TimeReturn, _name='time_return', fund=False)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown', fund=False)
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trade_analyzer')

    # 运行策略
    results = cerebro.run()
    strat = results[0]

    # 获取分析器的结果
    sharpe_ratio = strat.analyzers.sharpe_ratio.get_analysis()
    annual_returns = strat.analyzers.annual_returns.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    time_return = strat.analyzers.time_return.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
    trade_info = {
        '总交易数': trade_analyzer.total.total,
        '未平仓交易数': trade_analyzer.total.open,
        '已平仓交易数': trade_analyzer.total.closed,
        '当前连赢': trade_analyzer.streak.won.current,
        '最长连赢': trade_analyzer.streak.won.longest,
        '当前连输': trade_analyzer.streak.lost.current,
        '最长连输': trade_analyzer.streak.lost.longest,
        '总毛收益': trade_analyzer.pnl.gross.total,
        '平均毛收益': trade_analyzer.pnl.gross.average,
        '总净收益': trade_analyzer.pnl.net.total,
        '平均净收益': trade_analyzer.pnl.net.average,
        '获胜交易数': trade_analyzer.won.total,
        '总获胜收益': trade_analyzer.won.pnl.total,
        '平均获胜收益': trade_analyzer.won.pnl.average,
        '最大获胜收益': trade_analyzer.won.pnl.max,
        '失败交易数': trade_analyzer.lost.total,
        '总失败收益': trade_analyzer.lost.pnl.total,
        '平均失败收益': trade_analyzer.lost.pnl.average,
        '最大失败收益': trade_analyzer.lost.pnl.max,
        '多头交易数': trade_analyzer.long.total,
        '空头交易数': trade_analyzer.short.total,
        '多头总毛收益': trade_analyzer.long.pnl.total,
        '多头平均收益': trade_analyzer.long.pnl.average,
        '空头总毛收益': trade_analyzer.short.pnl.total,
        '空头平均收益': trade_analyzer.short.pnl.average,
        '胜率': f"{trade_analyzer.won.total / trade_analyzer.total.closed * 100 if trade_analyzer.total.closed > 0 else 0}%"
    }
    print(f"交易信息:\n{trade_info}")
    print(f"夏普比率: {sharpe_ratio['sharperatio']}")
    print(f"年化收益率: {annual_returns}")
    print(f"总收益率: {returns}")
    # print(f"时间收益: {time_return}")
    print(f"最大回撤: {drawdown}")

    # 设置图形参数并绘图
    import matplotlib.pyplot as plt

    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 1.0
    plt.rcParams['agg.path.chunksize'] = 10000
    cerebro.plot(style='candlestick', barup='black', bardown='white', marker='o', markersize=4, markercolor='orange')

    # 获取附图数据
    strat_analysis = strat.get_analysis()
    main_plot_dates = [bt.num2date(date) for date in strat.datas[0].datetime.array]

    # 绘制预测价格在附图中
    plt.figure()
    plt.plot(main_plot_dates, strat_analysis, color='blue', label='Predicted Price')
    plt.legend()
    plt.show()
#
# if __name__ == '__main__':
#     cerebro = bt.Cerebro()
#
#     # 为不同的时间帧添加数据
#     timeframes = ['5m', '15m', '30m']
#     data_feeds = []
#     for i, tf in enumerate(timeframes):
#         ohlcv_df = fetch_data('BNB/USDT', tf, 500)
#         data = bt.feeds.PandasData(dataname=ohlcv_df)
#         if tf == '5m':
#             data.plotinfo.plot = True  # 只有5分钟数据显示在主图
#         else:
#             data.plotinfo.plot = False  # 其他时间帧不显示或显示在子图
#             data.plotinfo.subplot = True
#         data_feeds.append(data)
#         cerebro.adddata(data)
#
#     cerebro.addstrategy(MultiTimeFrameRSIStrategy)
#     cerebro.broker.setcash(100000.0)
#
#     # 运行策略
#     cerebro.run()
#     plt.rcParams['path.simplify'] = True
#     plt.rcParams['path.simplify_threshold'] = 1.0
#     plt.rcParams['agg.path.chunksize'] = 10000
#     cerebro.plot(style='candlestick', barup='black', bardown='white', marker='o', markersize=4, markercolor='orange')

#
# def calculate_rsi(data, window=14):
#     rsi = RSIIndicator(data['close'], window=window).rsi()
#     return rsi
#
# # 获取数据
# data_5m = fetch_data('PEOPLE/USDT', '5m', 500)
# data_15m = fetch_data('PEOPLE/USDT', '15m', 500)
# data_30m = fetch_data('PEOPLE/USDT', '30m', 500)
#
# # 计算RSI
# data_5m['RSI'] = calculate_rsi(data_5m)
# data_15m['RSI'] = calculate_rsi(data_15m)
# data_30m['RSI'] = calculate_rsi(data_30m)
#
# # 同步索引
# index_5m = data_5m.index
# data_15m = data_15m.reindex(index_5m, method='nearest')
# data_30m = data_30m.reindex(index_5m, method='nearest')
#
#
# # 初始化买卖信号和冷却计数器
# buy_threshold = 30
# sell_threshold = 72
# cooldown_period = 10
# buy_cooldown = sell_cooldown = 0
#
# signals = {'Buy': [], 'Sell': []}
#
# for time in index_5m:
#     rsi5 = data_5m.at[time, 'RSI']
#     rsi15 = data_15m.at[time, 'RSI']
#     rsi30 = data_30m.at[time, 'RSI']
#
#     if buy_cooldown > 0:
#         buy_cooldown -= 1
#     if sell_cooldown > 0:
#         sell_cooldown -= 1
#
#     if rsi5 < buy_threshold and rsi15 < buy_threshold and rsi30 < buy_threshold and buy_cooldown == 0:
#         print(f"Buy signal at {time},price:{data_5m.at[time, 'close']}")
#         signals['Buy'].append(data_5m.at[time, 'close'])
#         signals['Sell'].append(np.nan)
#         buy_cooldown = cooldown_period
#     elif rsi5 > sell_threshold and rsi15 > sell_threshold and rsi30 > sell_threshold and sell_cooldown == 0:
#         print(f"Sell signal at {time},price:{data_5m.at[time, 'close']}")
#         signals['Sell'].append(data_5m.at[time, 'close'])
#         signals['Buy'].append(np.nan)
#         sell_cooldown = cooldown_period
#     else:
#         signals['Buy'].append(np.nan)
#         signals['Sell'].append(np.nan)
#
# # 绘图
# plt.figure(figsize=(30, 15))
# import matplotlib.dates as mdates
#
# ax = plt.gca()  # Get the current axes
#
# # 使用AutoDateLocator来自动选择日期间隔，同时设置MaxNLocator作为备份确保至少有16个主刻度
# date_locator = mdates.AutoDateLocator(minticks=40, maxticks=60)
# ax.xaxis.set_major_locator(date_locator)
# # 设置日期格式
# date_formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')  # 根据需要调整日期格式
# ax.xaxis.set_major_formatter(date_formatter)
#
# plt.plot(data_5m.index, data_5m['close'], label='Close Price', color='blue')
# plt.scatter(data_5m.index, signals['Buy'], color='green', label='Buy Signal', marker='^', s=100)
# plt.scatter(data_5m.index, signals['Sell'], color='red', label='Sell Signal', marker='v', s=100)
# plt.title('Price and Buy/Sell Signals on 5m Data')
#
# # 由于我们已经通过MaxNLocator设置了至少的刻度数量，可能不需要旋转标签了，但根据实际情况调整
# plt.xticks(rotation=45)
#
# plt.legend()
# plt.tight_layout()
# plt.show()








#
#
#
#
#
# timestamp = int(time.time() * 1000)  # 以毫秒为单位
#
# account_info = client.get_account(timestamp=timestamp)
#
#
#
#
#
#
#
#
# # 打印账户信息
# print(account_info)
# # 遍历账户的资产余额
# for balance in account_info["balances"]:
#     pair = balance["asset"]
#     free_balance = balance["free"]
#     print(f"交易对: {pair}, 免费余额: {free_balance}")




# # 指定你想要查询的交易对
# symbol = "BTCUSDT"  # 例如，你可以选择BTCUSDT或其他你感兴趣的交易对
#
# # 查询最近的交易记录
# trades = client.futures_recent_trades(symbol=symbol)
#
# # 打印交易记录
# print("交易记录:")
# for trade in trades:
#     print(trade)
#
# # 定义下单参数
# symbol = 'BTCUSDT'  # 交易对
# side = 'BUY'       # 买入
# type = 'LIMIT'     # 限价单
# # type = 'MARKET'     # 市价单
# timeInForce = 'GTC'  # 一直有效，直到成交或取消
# quantity = 0.001   # 购买数量，根据实际情况调整
# price = '50000'    # 期望购买价格，单位为USDT，根据市场情况调整
#
#
# # 创建订单
# try:
#     params = {
#         'symbol': symbol,
#         'side': side,
#         'type': type,
#         'quantity': quantity,
#         'price': price,
#         'timeInForce':timeInForce
#     }
#     order_response = client.create_order(**params)
#
#     print("Market order created successfully.",end='\n')
#     print(order_response)
# except Exception as e:
#     print(f"Failed to create market order: {e}",end="\n")