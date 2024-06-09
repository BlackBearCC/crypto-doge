import asyncio
import json

import ccxt.pro
import numpy as np


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




# 初始化CCXT的OKX接口
exchange = ccxt.okx({
    'enableRateLimit': True,
    'countries': ['CN'],
})
def fetch_data(symbol, timeframe, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_rsi(data, window=14):
    rsi = RSIIndicator(data['close'], window=window).rsi()
    return rsi

# 获取数据
data_5m = fetch_data('PEOPLE/USDT', '5m', 500)
data_15m = fetch_data('PEOPLE/USDT', '15m', 500)
data_30m = fetch_data('PEOPLE/USDT', '30m', 500)

# 计算RSI
data_5m['RSI'] = calculate_rsi(data_5m)
data_15m['RSI'] = calculate_rsi(data_15m)
data_30m['RSI'] = calculate_rsi(data_30m)

# 同步索引
index_5m = data_5m.index
data_15m = data_15m.reindex(index_5m, method='nearest')
data_30m = data_30m.reindex(index_5m, method='nearest')


# 初始化买卖信号和冷却计数器
buy_threshold = 30
sell_threshold = 72
cooldown_period = 10
buy_cooldown = sell_cooldown = 0

signals = {'Buy': [], 'Sell': []}

for time in index_5m:
    rsi5 = data_5m.at[time, 'RSI']
    rsi15 = data_15m.at[time, 'RSI']
    rsi30 = data_30m.at[time, 'RSI']

    if buy_cooldown > 0:
        buy_cooldown -= 1
    if sell_cooldown > 0:
        sell_cooldown -= 1

    if rsi5 < buy_threshold and rsi15 < buy_threshold and rsi30 < buy_threshold and buy_cooldown == 0:
        print(f"Buy signal at {time},price:{data_5m.at[time, 'close']}")
        signals['Buy'].append(data_5m.at[time, 'close'])
        signals['Sell'].append(np.nan)
        buy_cooldown = cooldown_period
    elif rsi5 > sell_threshold and rsi15 > sell_threshold and rsi30 > sell_threshold and sell_cooldown == 0:
        print(f"Sell signal at {time},price:{data_5m.at[time, 'close']}")
        signals['Sell'].append(data_5m.at[time, 'close'])
        signals['Buy'].append(np.nan)
        sell_cooldown = cooldown_period
    else:
        signals['Buy'].append(np.nan)
        signals['Sell'].append(np.nan)

# 绘图
plt.figure(figsize=(30, 15))
import matplotlib.dates as mdates

ax = plt.gca()  # Get the current axes

# 使用AutoDateLocator来自动选择日期间隔，同时设置MaxNLocator作为备份确保至少有16个主刻度
date_locator = mdates.AutoDateLocator(minticks=40, maxticks=60)
ax.xaxis.set_major_locator(date_locator)
# 设置日期格式
date_formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')  # 根据需要调整日期格式
ax.xaxis.set_major_formatter(date_formatter)

plt.plot(data_5m.index, data_5m['close'], label='Close Price', color='blue')
plt.scatter(data_5m.index, signals['Buy'], color='green', label='Buy Signal', marker='^', s=100)
plt.scatter(data_5m.index, signals['Sell'], color='red', label='Sell Signal', marker='v', s=100)
plt.title('Price and Buy/Sell Signals on 5m Data')

# 由于我们已经通过MaxNLocator设置了至少的刻度数量，可能不需要旋转标签了，但根据实际情况调整
plt.xticks(rotation=45)

plt.legend()
plt.tight_layout()
plt.show()








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