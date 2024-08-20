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

        # logging.info(f"余额 (Balances):")
        # for balance in account_info['balances']:
        #     logging.info(f"    资产 (asset): {balance['asset']}, 可用余额 (free): {balance['free']}, 冻结余额 (locked): {balance['locked']}")
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


def view_open_orders(client, symbol=None):
    """查看未完成的订单"""
    try:
        # 获取当前未完成的订单
        open_orders = client.get_open_orders(symbol=symbol)

        if open_orders:
            logging.info(f"当前未完成订单 (Open Orders) ({len(open_orders)}):")
            for order in open_orders:
                logging.info(
                    f"订单ID: {order['orderId']}, 交易对: {order['symbol']}, 数量: {order['origQty']}, 价格: {order['price']}, 状态: {order['status']}")
            return open_orders
        else:
            logging.info("没有未完成的订单。")
            return []

    except Exception as e:
        logging.error(f"获取订单失败: {str(e)}")
        return []



def cancel_order(client, symbol, order_id):
    """取消指定订单"""
    try:
        cancel_response = client.cancel_order(symbol=symbol, orderId=order_id)
        logging.info(f"已取消订单: {cancel_response}")
        return cancel_response
    except Exception as e:
        logging.error(f"取消订单失败: {str(e)}")
        return None

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

open_orders = view_open_orders(client)


def view_order_history(client, symbol, limit=100):
    """查看订单历史记录"""
    try:
        # 获取订单历史记录
        orders = client.get_all_orders(symbol=symbol, limit=limit)

        if orders:
            logging.info(f"订单历史记录 (Order History) for {symbol}:")
            for order in orders:
                logging.info(
                    f"订单ID: {order['orderId']}, 交易对: {order['symbol']}, 价格: {order['price']}, 数量: {order['origQty']}, 状态: {order['status']}, 类型: {order['type']}, 时间: {order['time']}")
            return orders
        else:
            logging.info(f"{symbol} 没有订单历史记录。")
            return []

    except Exception as e:
        logging.error(f"获取订单历史记录失败: {str(e)}")
        return []


# 示例用法，查看BTCUSDT交易对的订单记录
# order_history = view_order_history(client, symbol="BTCUSDT", limit=50)

def sell_btc():
    """卖出BTC"""
    try:
        # 获取账户信息，查找BTC余额
        account_info = client.get_account()
        btc_balance = None

        for balance in account_info['balances']:
            if balance['asset'] == 'BTC':
                btc_balance = float(balance['free'])  # 获取可用余额
                break

        if btc_balance and btc_balance > 0:
            # 创建市价卖单，将所有BTC卖出换成USDT
            order = client.create_order(
                symbol='BTCUSDT',
                side='SELL',
                type='MARKET',
                quantity=btc_balance
            )
            logging.info(f"已创建市价卖单: {order}")
        else:
            logging.info("没有可用的BTC余额。")

    except Exception as e:
        logging.error(f"卖出BTC失败: {str(e)}")



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
# print(f"账户信息：{account_info}")


def view_trade_history(client, symbol, limit=100):
    """查看交易历史记录"""
    try:
        # 获取交易历史记录
        trades = client.get_my_trades(symbol=symbol, limit=limit)

        if trades:
            logging.info(f"交易历史记录 (Trade History) for {symbol}:")
            for trade in trades:
                logging.info(
                    f"交易ID: {trade['id']}, 订单ID: {trade['orderId']}, 价格: {trade['price']}, 数量: {trade['qty']}, 佣金: {trade['commission']}, 佣金资产: {trade['commissionAsset']}, 时间: {trade['time']}")
            return trades
        else:
            logging.info(f"{symbol} 没有历史交易记录。")
            return []

    except Exception as e:
        logging.error(f"获取交易历史记录失败: {str(e)}")
        return []


# 示例用法，查看BTCUSDT交易对的历史记录
# trade_history = view_trade_history(client, symbol="BTCUSDT", limit=50)
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


def plot_predictions(historical_datetimes, historical_prices, future_datetimes, future_prices, peaks, valleys):
    """绘制历史数据（蓝色）和预测数据（黄色）以及波峰波谷"""
    plt.figure(figsize=(15, 5))

    # 绘制实际的历史数据
    plt.plot(historical_datetimes, historical_prices, color='blue', lw=2, label='Historical Prices')

    # 绘制预测的未来数据
    plt.plot(future_datetimes, future_prices, color='orange', lw=2, label='Predicted Prices')

    # 合并历史和预测数据以标记波峰和波谷
    combined_datetimes = np.concatenate((historical_datetimes, future_datetimes))
    combined_prices = np.concatenate((historical_prices, future_prices))

    plt.plot([combined_datetimes[i] for i in peaks], [combined_prices[i] for i in peaks], "x", color='green', label='Peaks')
    plt.plot([combined_datetimes[i] for i in valleys], [combined_prices[i] for i in valleys], "o", color='red', label='Valleys')

    plt.title('Historical and Predicted Prices with Peaks and Valleys')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def wait_until_next_hour():
    """等待直到下一个整点,该函数计算当前时间与下一个整点的时间差，并休眠这段时间，确保代码在整点运行。"""
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
    sleep_time = (next_hour - now).total_seconds()
    logging.info(f"等待 {sleep_time} 秒，直到下一个整点 {next_hour.strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(sleep_time)
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

        # 提取历史数据
        historical_prices = [d['close'] for d in data_history[-trend_window:]]
        historical_datetimes = pd.date_range(data_history[-trend_window]['open_time'], periods=trend_window, freq='H')

        # 合并历史数据和预测数据进行波峰波谷识别
        combined_prices = np.concatenate((historical_prices, future_prices))
        combined_datetimes = np.concatenate((historical_datetimes, future_datetimes))
        peaks, valleys = find_peaks(combined_prices)[0], find_peaks(-combined_prices)[0]

        # 绘制历史和预测的价格以及波峰波谷
        plot_predictions(historical_datetimes, historical_prices, future_datetimes, future_prices, peaks, valleys)

        # 获取最近的收盘价
        current_price = data_history[-1]['close']
        current_time = data_history[-1]['open_time']

        # 查找最近的波谷和波峰
        recent_valley_index = valleys[-1] if len(valleys) > 0 else None
        recent_peak_index = peaks[-1] if len(peaks) > 0 else None

        if recent_valley_index is not None:
            recent_valley_price = combined_prices[recent_valley_index]
            recent_valley_time = combined_datetimes[recent_valley_index]

        if recent_peak_index is not None:
            recent_peak_price = combined_prices[recent_peak_index]
            recent_peak_time = combined_datetimes[recent_peak_index]
            # 判断当前价格是否处于最近的波谷时间区间内
            if recent_valley_index is not None and recent_valley_time <= current_time <= (
                    recent_valley_time + timedelta(hours=1)):
                if current_inventory == 0 and cash >= buy_amount:
                    logging.info(f"当前价格 {current_price} 处于波谷区间，执行买入操作。")
                    buy_price = current_price
                    buy_units = buy_amount / buy_price
                    cash -= buy_amount
                    current_inventory += buy_units
                    states_buy.append(len(data_history) - 1)
                    trades.append({
                        'datetime': current_time,
                        'price': buy_price,
                        'size': buy_units,
                        'action': 'buy'
                    })
                    execute_trade(client, symbol, "BUY", buy_units)
            else:
                logging.info("当前价格不在波谷区间，未执行交易。")

        current_portfolio_value = cash + current_inventory * data_history[-1]['close']
        portfolio_value.append(current_portfolio_value)
        logging.info("Current Portfolio Value: %.2f", current_portfolio_value)

        # 记录账户信息
        log_account_info(client)

        # 等待直到下一个整点
        wait_until_next_hour()


run_strategy()