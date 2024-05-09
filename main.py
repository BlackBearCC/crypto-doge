import ccxt
import pandas as pd
import backtrader as bt
import backtrader.feeds as btfeeds
import datetime
# 使用ccxt获取数据
exchange = ccxt.okx()
symbol = 'BTC/USDT'
timeframe = '1d'
since = exchange.parse8601('2020-01-01T00:00:00Z')
limit = 100
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

# 转换为Pandas DataFrame
df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)
df['openinterest'] = 0  # 添加openinterest列

fromdate = df.index.min()
todate = df.index.max()

# 使用计算得出的日期范围创建backtrader数据源
data = bt.feeds.PandasData(dataname=df, fromdate=fromdate, todate=todate)



# 定义策略
class MACDCrossStrategy(bt.Strategy):
    params = (
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.params.macd1,
                                       period_me2=self.params.macd2,
                                       period_signal=self.params.macdsig)
        self.cross_up = bt.indicators.CrossOver(self.macd.macd - self.macd.signal, self.macd.signal)
        self.cross_down = bt.indicators.CrossOver(self.macd.signal, self.macd.macd - self.macd.signal)

    def next(self):
        if not self.position:
            if self.cross_up[0] > 0:
                self.buy()
        else:
            if self.cross_down[0] > 0:
                self.close()


# 创建Cerebro实例
cerebro = bt.Cerebro()

# # 添加数据
# data = PandasData()
cerebro.adddata(data)

# 添加策略
cerebro.addstrategy(MACDCrossStrategy)

# 设置初始资本
cerebro.broker.setcash(10000.0)

# 设置佣金
cerebro.broker.setcommission(commission=0.001)

cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.SharpeRatio)
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

# # 执行回测
# cerebro.run()

# 运行策略
results = cerebro.run()
strat = results[0]

# 获取最终组合价值
# 获取并打印关键数据
print('最终资产: %.2f' % cerebro.broker.getvalue())
print('最大回撤: %.2f%%' % (strat.analyzers.drawdown.get_analysis()['max']['drawdown']))
print('年化收益率: %.2f%%' % (strat.analyzers.returns.get_analysis()['rnorm100']))
print('夏普比率: %.2f' % (strat.analyzers.sharperatio.get_analysis()['sharperatio']))

# 绘制
# cerebro.plot()
