import numpy as np
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import backtrader as bt
import datetime

# 加载交易所数据
exchange = ccxt.okx()
symbol = 'BTC/USDT'
timeframe = '1d'
since = exchange.parse8601('2024-01-01T00:00:00Z')
limit = 500  # 取最近500天的数据
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# 计算MACD
def calculate_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['signal'] = signal
    df['hist'] = macd - signal

calculate_macd(df)

# 寻找买入信号的背离点
def detect_buy_signals(df):
    buy_signals = []
    for i in range(1, len(df)):
        if df['hist'][i] > df['hist'][i-1]:  # 简化的示例，真实策略需要更复杂的逻辑
            buy_signals.append(df['timestamp'][i])
    return buy_signals

buy_signals = detect_buy_signals(df)
# 绘制收盘价和MACD指标
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 绘制价格
ax1.plot(df['timestamp'], df['close'], label='Close Price')
ax1.set_title('Price and MACD Indicator')
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')

# 在买入信号处标记
for signal in buy_signals:
    ax1.plot(df.loc[df['timestamp'] == signal, 'timestamp'],
             df.loc[df['timestamp'] == signal, 'close'],
             '^', markersize=10, color='g', lw=0, label='Buy Signal')

# 绘制MACD和信号线
ax2.plot(df['timestamp'], df['macd'], label='MACD', color='blue')
ax2.plot(df['timestamp'], df['signal'], label='Signal Line', color='orange')
ax2.bar(df['timestamp'], df['hist'], label='Histogram', color='grey', width=0.7)
ax2.set_ylabel('MACD')
ax2.legend(loc='upper left')

plt.show()

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(window=window).mean()

calculate_atr(df)

# 初始化
initial_capital = 1000.0
current_capital = initial_capital
peak_capital = initial_capital  # 账户价值的最高点
max_drawdown = 0  # 最大资金回撤
position = 0
buy_price = 0
stop_loss = 0
take_profit = 0

# 在回测前初始化账户总值历史记录列表
capital_history = []
# 回测
for i in range(1, len(df)):
    current_price = df['close'][i]
    atr = df['atr'][i]

    # 更新最大回撤
    if current_capital > peak_capital:
        peak_capital = current_capital
    else:
        drawdown = (peak_capital - current_capital) / peak_capital
        max_drawdown = max(max_drawdown, drawdown)

    # 检测买入信号
    if df['timestamp'][i] in buy_signals and current_capital > 0:
        # 计算止损和止盈点
        buy_price = current_price
        stop_loss = buy_price - atr
        take_profit = buy_price + 1.5 * atr
        position = current_capital / buy_price
        current_capital = 0  # 投入所有资金买入

    # 检测止损或止盈条件
    if position > 0 and (current_price <= stop_loss or current_price >= take_profit):
        # 卖出
        current_capital = position * current_price
        position = 0  # 清空持仓

        # 每次更新current_capital后，添加到capital_history
    if position > 0:
        capital_history.append(position * current_price)
    else:
        capital_history.append(current_capital)

# 计算最终价值和添加到历史记录
if position > 0:
    current_capital = position * df['close'].iloc[-1]
capital_history.append(current_capital)



# 更新最大回撤
final_drawdown = (peak_capital - current_capital) / peak_capital
max_drawdown = max(max_drawdown, final_drawdown)

return_rate = (current_capital - initial_capital) / initial_capital * 100

print(f"最终资金: {current_capital:.2f} USD")
print(f"收益率: {return_rate:.2f}%")

def calculate_max_drawdown(capital_history):
    """
    计算最大资金回撤
    :param capital_history: 账户总值的时间序列，列表或一维数组形式
    :return: 最大资金回撤的百分比
    """
    peak = capital_history[0]
    max_drawdown = 0
    for capital in capital_history:
        peak = max(peak, capital)
        drawdown = (peak - capital) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown * 100  # 转换为百分比



# 使用实际的资金历史来计算最大资金回撤
max_drawdown_percentage = calculate_max_drawdown(capital_history)
print(f"最大资金回撤: {max_drawdown_percentage:.2f}%")