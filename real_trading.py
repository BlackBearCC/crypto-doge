import os
import time
import numpy as np
import pandas as pd
from binance.client import Client
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus.para import Paragraph
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import logging

# 设置日志记录，并添加控制台输出
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('trading_log.log'),  # 写入文件
                        logging.StreamHandler()  # 输出到控制台
                    ])
def log_account_info(client):
    """记录账户信息"""
    try:
        account_info = client.get_account()
        logging.info("账户信息 (Account Info):")
        logging.info(f"手续费 (makerCommission): {account_info['makerCommission']}")
        logging.info(f"手续费 (takerCommission): {account_info['takerCommission']}")
        logging.info(f"买方手续费 (buyerCommission): {account_info['buyerCommission']}")
        logging.info(f"卖方手续费 (sellerCommission): {account_info['sellerCommission']}")
        logging.info(f"可交易 (canTrade): {account_info['canTrade']}")
        logging.info(f"可提取 (canWithdraw): {account_info['canWithdraw']}")
        logging.info(f"可存入 (canDeposit): {account_info['canDeposit']}")

        logging.info(f"余额 (Balances):")
        for balance in account_info['balances']:
            logging.info(f"    资产 (asset): {balance['asset']}, 可用余额 (free): {balance['free']}, 冻结余额 (locked): {balance['locked']}")
        logging.info("")

        # 打印账户的订单信息
        open_orders = client.get_open_orders()
        logging.info("当前未完成订单 (Open Orders):")
        for order in open_orders:
            log_order_details(order)

    except Exception as e:
        logging.error("获取账户信息失败: %s", str(e))


def log_order_details(order):
    """将订单详细信息以可读的中文格式记录到日志"""
    logging.info("订单详情 (order details):")
    logging.info(f"交易对 (symbol): {order['symbol']}")
    logging.info(f"订单ID (orderId): {order['orderId']}")
    logging.info(f"客户订单ID (clientOrderId): {order['clientOrderId']}")
    logging.info(f"交易时间 (transactTime): {order['transactTime']}")
    logging.info(f"价格 (price): {order['price']}")
    logging.info(f"原始数量 (origQty): {order['origQty']}")
    logging.info(f"已执行数量 (executedQty): {order['executedQty']}")
    logging.info(f"累计成交金额 (cummulativeQuoteQty): {order['cummulativeQuoteQty']}")
    logging.info(f"订单状态 (status): {order['status']}")
    logging.info(f"订单类型 (type): {order['type']}")
    logging.info(f"订单方向 (side): {order['side']}")
    logging.info("成交明细 (fills):")
    for fill in order['fills']:
        logging.info(f"    成交价格 (price): {fill['price']}, 成交数量 (qty): {fill['qty']}, 佣金 (commission): {fill['commission']}, 佣金资产 (commissionAsset): {fill['commissionAsset']}")
    logging.info(f"自成交预防模式 (selfTradePreventionMode): {order.get('selfTradePreventionMode', '无')}")
    logging.info("")

# Binance API 配置
api_key = "7XbBmjA1UxBzNBe0AriKyYlwt2HvOlNEzftJ9bN2g5kbUFACDKppATNlqGBtvlNE"
api_secret = "2BLZojVtSzDfyVgE1TW6U6MCSxDoDh5pnNZnz0BohEOGc7duHsT7mob2jf42ksOA"
client = Client(api_key, api_secret, testnet=True)

# 加载模型
model_path = 'quant_model.h5'
modelnn = tf.keras.models.load_model(model_path)
logging.info("模型已加载: %s", model_path)

# 初始化参数
timestamp = 5  # 时间步长
initial_money = 10000
buy_amount = 1000
max_sell = 10
stop_loss = 0.03
take_profit = 0.07
trend_window = 14  # 预测窗口长度
minmax = MinMaxScaler()

cash = initial_money
current_inventory = 0
states_buy = []
states_sell = []
portfolio_value = [cash]
data_history = []
trades = []
predicted_prices = []
trend_list = []
market_states = []
future_prices_all = []
peaks_all = []
valleys_all = []
future_datetimes_all = []
processed_trades = set()

symbol = "BTCUSDT"  # 交易对

force_trade = True  # 设置为True以强制触发交易，方便测试
account_info = client.get_account()
# 获取交易对的精度信息
def get_symbol_info(client, symbol):
    info = client.get_symbol_info(symbol)
    print(info)
    step_size = None
    for f in info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            step_size = float(f['stepSize'])
            break
    if step_size is None:
        raise ValueError("No stepSize found for symbol.")
    return step_size
step_size = get_symbol_info(client, symbol)  # 获取交易对的精度限制
logging.info("交易对精度信息：%s", step_size)
print(f"账户信息：{account_info}")
def get_realtime_data(client, symbol, interval='1h', lookback='48'):
    """从Binance获取实时数据"""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    # 时区转换
    df['open_time'] = df['open_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


def predict_future(last_data, modelnn, minmax, timestamp, future_hours=14):
    """预测未来价格"""
    future_predictions = []
    close_price = last_data[-timestamp:]
    # 使用历史数据进行fit
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


def identify_trend(future_predictions):
    """识别价格趋势"""
    peaks, _ = find_peaks(future_predictions)
    valleys, _ = find_peaks(-np.array(future_predictions))

    trends = ['sideways'] * len(future_predictions)
    trend_points = sorted(peaks.tolist() + valleys.tolist())

    for i in range(1, len(trend_points)):
        start = trend_points[i - 1]
        end = trend_points[i]
        if future_predictions[end] > future_predictions[start]:
            trends[start:end + 1] = ['uptrend'] * (end - start + 1)
        else:
            trends[start:end + 1] = ['downtrend'] * (end - start + 1)

    return trends, peaks, valleys

def round_step_size(quantity, step_size):
    """将数量四舍五入到正确的精度"""
    return round(quantity - (quantity % step_size), 8)
def execute_trade(client, symbol, side, quantity, price=None, type='MARKET'):
    """执行交易"""
    try:
        # 将数量四舍五入到正确的精度
        quantity = round_step_size(quantity, step_size)
        # 如果是市价单
        if type == 'MARKET':
            order = client.create_order(
                symbol=symbol,
                side=side,
                type=type,
                quantity=quantity
            )
        # 如果是限价单
        elif type == 'LIMIT' and price:
            order = client.create_order(
                symbol=symbol,
                side=side,
                type=type,
                timeInForce='GTC',
                quantity=quantity,
                price=str(price)
            )
        else:
            raise ValueError("Invalid order type or missing price for LIMIT order.")

            # 记录订单详细信息
        log_order_details(order)
        return order
    except Exception as e:
        logging.error("Failed to place %s order: %s", side, str(e))
        return None


def plot_predictions(future_datetimes, future_prices, peaks, valleys):
    """绘制预测的价格和波峰波谷"""
    plt.figure(figsize=(15, 5))
    plt.plot(future_datetimes, future_prices, color='orange', lw=2, label='Predicted Prices')

    plt.plot([future_datetimes[i] for i in peaks], [future_prices[i] for i in peaks], "x", color='green', label='Peaks')
    plt.plot([future_datetimes[i] for i in valleys], [future_prices[i] for i in valleys], "o", color='red',
             label='Valleys')

    plt.title('Predicted Future Prices with Peaks and Valleys')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def run_strategy():
    global cash, current_inventory, states_buy, states_sell, portfolio_value, data_history, trades

    logging.info("开始运行策略...")
    # 获取实时数据
    df = get_realtime_data(client, symbol)
    data_history = df.reset_index().to_dict('records')  # 使用 reset_index 保留 'open_time'
    logging.info("获取实时数据成功:" + str(data_history))

    while True:
        # 每小时获取一次新数据
        new_data = get_realtime_data(client, symbol, lookback='1')
        new_data_record = new_data.reset_index().to_dict('records')[0]  # 保留 'open_time'
        data_history.append(new_data_record)

        # 进行未来指定窗口的预测
        future_prices = predict_future([d['close'] for d in data_history[-timestamp:]], modelnn, minmax, timestamp,
                                       future_hours=trend_window)
        # 计算预测的未来时间
        future_datetimes = pd.date_range(data_history[-1]['open_time'] + timedelta(hours=1), periods=trend_window,
                                         freq='H')

        # 记录预测的价格和对应的未来时间到日志
        future_prices_with_time = list(zip(future_datetimes, future_prices))
        logging.info("Predicted future prices with time: %s", future_prices_with_time)

        # 识别趋势和波峰波谷
        trends, peaks, valleys = identify_trend(future_prices)

        # 绘制预测结果和波峰波谷
        future_datetimes = pd.date_range(data_history[-1]['open_time'] + timedelta(hours=1), periods=trend_window,
                                         freq='H')
        plot_predictions(future_datetimes, future_prices, peaks, valleys)
        force_trade=True
        # 测试交易触发条件
        if force_trade:
            print("Force trading for testing purposes...")

            if current_inventory == 0 and cash >= buy_amount:
                buy_price = data_history[-1]['close']
                buy_units = buy_amount / buy_price
                cash -= buy_amount
                current_inventory += buy_units
                states_buy.append(len(data_history) - 1)
                trades.append({
                    'datetime': data_history[-1]['open_time'],
                    'price': buy_price,
                    'size': buy_units,
                    'action': 'buy'
                })
                execute_trade(client, symbol, "BUY", buy_units)

            elif current_inventory > 0:
                sell_price = data_history[-1]['close']
                sell_units = min(current_inventory, max_sell)
                cash += sell_units * sell_price
                current_inventory -= sell_units
                states_sell.append(len(data_history) - 1)
                trades.append({
                    'datetime': data_history[-1]['open_time'],
                    'price': sell_price,
                    'size': sell_units,
                    'action': 'sell'
                })
                execute_trade(client, symbol, "SELL", sell_units)

            # 只触发一次测试交易
            force_trade = False
        # 处理买卖信号
        for i in range(min(len(valleys), len(peaks))):
            if peaks[i] > valleys[i]:  # 波峰必须在波谷之后
                actual_valley_index = len(data_history) - trend_window + valleys[i]
                actual_peak_index = len(data_history) - trend_window + peaks[i]

                if (actual_valley_index, actual_peak_index) in processed_trades:
                    continue

                buy_price = data_history[-(trend_window - valleys[i])]['close']
                sell_price = data_history[-(trend_window - peaks[i])]['close']
                if abs(sell_price - buy_price) < 800:
                    continue

                logging.info("Detected valley at index %d, predicted buy price: %.2f, actual: %.2f",
                             actual_valley_index, future_prices[valleys[i]], buy_price)
                logging.info("Detected peak at index %d, predicted sell price: %.2f, actual: %.2f", actual_peak_index,
                             future_prices[peaks[i]], sell_price)

                if current_inventory == 0 and cash >= buy_amount:
                    buy_units = buy_amount / buy_price
                    cash -= buy_amount
                    current_inventory += buy_units
                    states_buy.append(actual_valley_index)
                    trades.append({
                        'datetime': data_history[-1]['open_time'],
                        'price': buy_price,
                        'size': buy_units,
                        'action': 'buy'
                    })
                    execute_trade(client, symbol, "BUY", buy_units)

                if current_inventory > 0 and actual_peak_index > actual_valley_index:
                    sell_units = min(current_inventory, max_sell)
                    cash += sell_units * sell_price
                    current_inventory -= sell_units
                    states_sell.append(actual_peak_index)
                    trades.append({
                        'datetime': data_history[-1]['open_time'],
                        'price': sell_price,
                        'size': sell_units,
                        'action': 'sell'
                    })
                    execute_trade(client, symbol, "SELL", sell_units)

                    trade_profit = (sell_price - buy_price) * sell_units
                    logging.info("Trade profit: %.2f", trade_profit)

                processed_trades.add((actual_valley_index, actual_peak_index))
                break

        current_portfolio_value = cash + current_inventory * data_history[-1]['close']
        portfolio_value.append(current_portfolio_value)
        logging.info("Current Portfolio Value: %.2f", current_portfolio_value)

        # 记录账户信息
        log_account_info(client)
        time.sleep(3600)  # 等待1小时再获取新数据


run_strategy()