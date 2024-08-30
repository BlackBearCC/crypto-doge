import os
import time
import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime, timedelta
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('trading_log.log'),
                        logging.StreamHandler()
                    ])

# Binance API 配置
api_key = "7XbBmjA1UxBzNBe0AriKyYlwt2HvOlNEzftJ9bN2g5kbUFACDKppATNlqGBtvlNE"
api_secret = "2BLZojVtSzDfyVgE1TW6U6MCSxDoDh5pnNZnz0BohEOGc7duHsT7mob2jf42ksOA"
client = Client(api_key, api_secret, testnet=True)


# 加载模型
model_path = 'quant_model.h5'
modelnn = tf.keras.models.load_model(model_path)

# 初始化参数
timestamp = 5  # 时间步长
initial_money = 10000
buy_amount = 1000
max_sell = 10
trend_window = 14  # 预测窗口长度
minmax = MinMaxScaler()

symbol = "BTCUSDT"  # 交易对

def get_realtime_data(client, symbol, interval='1h', lookback='3000'):
    """从Binance获取实时数据"""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['open_time'] = df['open_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def predict_future(last_data, modelnn, minmax, timestamp, future_hours=14):
    """预测未来价格"""
    future_predictions = []
    close_price = last_data[-timestamp:]
    minmax.fit(np.array(close_price).reshape(-1, 1))
    close_price_scaled = minmax.transform(np.array(close_price).reshape(-1, 1)).flatten()

    for _ in range(future_hours):
        input_data = np.column_stack((np.zeros(timestamp),
                                      np.zeros(timestamp),
                                      np.zeros(timestamp),
                                      close_price_scaled))
        prediction = modelnn.predict(input_data[np.newaxis, :, :])
        predicted_close_price_scaled = prediction[0, -1]
        predicted_close_price = minmax.inverse_transform([[predicted_close_price_scaled]])[0, 0]
        future_predictions.append(predicted_close_price)

        close_price = np.append(close_price[1:], predicted_close_price)
        close_price_scaled = minmax.transform(np.array(close_price).reshape(-1, 1)).flatten()

    return future_predictions

def execute_trade(client, symbol, side, quantity):
    """执行交易"""
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        logging.info(f"执行{side}单: {order}")
        return order
    except Exception as e:
        logging.error(f"交易执行失败: {str(e)}")
        return None

def run_strategy():
    while True:
        # 获取实时数据
        df = get_realtime_data(client, symbol)
        
        # 进行未来指定窗口的预测
        future_prices = predict_future(df['close'].values, modelnn, minmax, timestamp, future_hours=trend_window)
        
        # 获取最近的收盘价
        current_price = df['close'].iloc[-1]
        
        # 计算RSI (这里需要添加RSI计算逻辑)
        rsi_5m = calculate_rsi(df, period=14)  # 假设使用14期RSI
        rsi_15m = calculate_rsi(df.resample('15T').last(), period=14)
        rsi_30m = calculate_rsi(df.resample('30T').last(), period=14)
        
        rsi_average = (rsi_5m.iloc[-1] + rsi_15m.iloc[-1] + rsi_30m.iloc[-1]) / 3
        
        # 动态阈值计算
        predicted_change = future_prices[-1] - current_price
        buy_threshold, sell_threshold = dynamic_threshold(predicted_change, 30, 70)
        
        # 交易逻辑
        if (rsi_5m.iloc[-1] < buy_threshold and rsi_15m.iloc[-1] < buy_threshold and 
            rsi_30m.iloc[-1] < buy_threshold and rsi_average < buy_threshold):
            # 执行买入
            quantity = buy_amount / current_price
            execute_trade(client, symbol, "BUY", quantity)
            logging.info(f"买入信号触发 - 价格: {current_price:.2f}, 数量: {quantity:.4f}")
        
        elif (rsi_5m.iloc[-1] > sell_threshold and rsi_15m.iloc[-1] > sell_threshold and 
              rsi_30m.iloc[-1] > sell_threshold and rsi_average > sell_threshold):
            # 执行卖出
            quantity = max_sell
            execute_trade(client, symbol, "SELL", quantity)
            logging.info(f"卖出信号触发 - 价格: {current_price:.2f}, 数量: {quantity:.4f}")
        
        # 等待一段时间再进行下一次检查
        time.sleep(300)  # 每5分钟检查一次

def calculate_rsi(data, period=14):
    """计算RSI"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def dynamic_threshold(predicted_change, base_buy_threshold, base_sell_threshold):
    """动态调整RSI阈值"""
    normalized_predicted_change = normalize(predicted_change, -2000, 2000)
    adjustment_factor = 50
    if normalized_predicted_change > 0:
        buy_threshold = base_buy_threshold + adjustment_factor * normalized_predicted_change
        sell_threshold = base_sell_threshold + adjustment_factor * normalized_predicted_change * 0.6
    else:
        buy_threshold = base_buy_threshold - adjustment_factor * normalized_predicted_change * 0.6
        sell_threshold = base_sell_threshold - adjustment_factor * normalized_predicted_change
    return buy_threshold, sell_threshold

def normalize(value, min_value, max_value):
    """归一化函数"""
    return 2 * (value - min_value) / (max_value - min_value) - 1

if __name__ == "__main__":
    run_strategy()