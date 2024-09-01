import os
import threading
import time
import numpy as np
import pandas as pd
from binance.client import Client
from prompt_toolkit import Application
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
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, CallbackContext

# 设置日志记录，并添加控制台输出
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('trading_log.log'),  # 写入文件
                        logging.StreamHandler()  # 输出到控制台
                    ])

# Telegram 机器人配置
TELEGRAM_TOKEN = '7516595825:AAHgdjEqJcGvOCojTs7-ZcMuW-G142157jg'
CHAT_ID = 'YOUR_CHAT_ID'
bot = Bot(token=TELEGRAM_TOKEN)
def send_telegram_message(message):
    """发送消息到Telegram"""
    bot.send_message(chat_id=CHAT_ID, text=message)

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

atr = 14

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
def get_realtime_data(client, symbol, interval='5m', lookback='3000'):
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


def predict_future(data_1h, modelnn, minmax, timestamp, future_hours=14):
    """预测未来价格，使用1小时数据"""
    # 获取最近的timestamp个1小时数据点
    last_data = data_1h['close'].values[-timestamp:]
    
    future_predictions = []
    minmax.fit(last_data.reshape(-1, 1))
    close_price_scaled = minmax.transform(last_data.reshape(-1, 1)).flatten()

    for _ in range(future_hours):
        input_data = np.column_stack((np.zeros(timestamp),
                                      np.zeros(timestamp),
                                      np.zeros(timestamp),
                                      close_price_scaled))
        prediction = modelnn.predict(input_data[np.newaxis, :, :])
        predicted_close_price_scaled = prediction[0, -1]
        predicted_close_price = minmax.inverse_transform([[predicted_close_price_scaled]])[0, 0]
        future_predictions.append(predicted_close_price)

        close_price_scaled = np.append(close_price_scaled[1:], predicted_close_price_scaled)

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
              # 设置止盈止损
        if side == 'BUY':
            stop_price = price - atr * atr_multiplier * 0.5
            take_profit_price = price + atr * atr_multiplier
            client.create_order(
                symbol=symbol,
                side='SELL',
                type='STOP_LOSS_LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                stopPrice=str(stop_price),
                price=str(stop_price)
            )
            client.create_order(
                symbol=symbol,
                side='SELL',
                type='TAKE_PROFIT_LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                stopPrice=str(take_profit_price),
                price=str(take_profit_price)
            )
        elif side == 'SELL':
            stop_price = price + atr * atr_multiplier * 0.5
            take_profit_price = price - atr * atr_multiplier
            client.create_order(
                symbol=symbol,
                side='BUY',
                type='STOP_LOSS_LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                stopPrice=str(stop_price),
                price=str(stop_price)
            )
            client.create_order(
                symbol=symbol,
                side='BUY',
                type='TAKE_PROFIT_LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                stopPrice=str(take_profit_price),
                price=str(take_profit_price)
            )
        
        # 发送交易信息到Telegram
        send_telegram_message(f"交易执行: {side} {quantity} {symbol} @ {price}")
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
def wait_until_next_5min():
    """等待直到下一个5分钟整点"""
    now = datetime.now()
    next_5min = now + timedelta(minutes=5 - now.minute % 5, seconds=-now.second, microseconds=-now.microsecond)
    sleep_time = (next_5min - now).total_seconds()
    logging.info(f"等待 {sleep_time:.2f} 秒，直到下一个5分钟整点 {next_5min.strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(sleep_time)
def wait_until_next_hour():
    """等待直到下一个整点小时"""
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
    sleep_time = (next_hour - now).total_seconds()
    logging.info(f"等待 {sleep_time} 秒，直到下一个整点 {next_hour.strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(sleep_time)

# 全局变量，用于存储最新的预测和阈值信息
latest_prediction_info = {}

def update_latest_prediction_info(current_time, current_price, predicted_change, rsi_5m, rsi_15m, rsi_30m, rsi_average, buy_threshold, sell_threshold):
    """更新最新的预测和阈值信息"""
    global latest_prediction_info
    latest_prediction_info = {
        "current_time": current_time,
        "current_price": current_price,
        "predicted_change": predicted_change,
        "rsi_5m": rsi_5m,
        "rsi_15m": rsi_15m,
        "rsi_30m": rsi_30m,
        "rsi_average": rsi_average,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold
    }

# 在记录详细的交易信息时，更新最新的预测和阈值信息
def log_trade_info(current_time, current_price, predicted_change, rsi_5m, rsi_15m, rsi_30m, rsi_average, buy_threshold, sell_threshold):
    """记录详细的交易信息"""
    logging.info(f"当前时间: {current_time}")
    logging.info(f"当前价格: {current_price:.2f}")
    logging.info(f"预测价格变化: {predicted_change:.2f}")
    logging.info(f"RSI (5分钟): {rsi_5m.iloc[-1]:.2f}")
    logging.info(f"RSI (15分钟): {rsi_15m.iloc[-1]:.2f}")
    logging.info(f"RSI (30分钟): {rsi_30m.iloc[-1]:.2f}")
    logging.info(f"RSI 平均值: {rsi_average:.2f}")
    logging.info(f"买入阈值: {buy_threshold:.2f}")
    logging.info(f"卖出阈值: {sell_threshold:.2f}")

    # 更新最新的预测和阈值信息
    update_latest_prediction_info(current_time, current_price, predicted_change, rsi_5m, rsi_15m, rsi_30m, rsi_average, buy_threshold, sell_threshold)

# 在 run_strategy 函数中调用 log_trade_info 而不是直接记录日志
def run_strategy():
   
    logging.info("开始运行策略...")
    # 获取实时数据
    df = get_realtime_data(client, symbol)
    data_history = df.reset_index().to_dict('records')  # 使用 reset_index 保留 'open_time'
    # logging.info("获取实时数据成功:" + str(data_history))
    last_prediction_time = None
    predicted_change = 0

    while True:
        # 等待到下一个5分钟整点
        wait_until_next_5min()
        
        current_time = datetime.now()
        
        # 获取实时数据
        df_5m = get_realtime_data(client, symbol, interval='5m')
        df_15m = get_realtime_data(client, symbol, interval='15m')
        df_30m = get_realtime_data(client, symbol, interval='30m')
        df_1h = get_realtime_data(client, symbol, interval='1h')
        
        # 检查是否需要进行新的预测（每整点小时一次）
        if last_prediction_time is None or current_time.minute == 0:
            future_prices = predict_future(df_1h, modelnn, minmax, timestamp, future_hours=trend_window)
            predicted_change = future_prices[-1] - df_1h['close'].iloc[-1]
            last_prediction_time = current_time
            logging.info(f"新的预测完成: 预测变化 = {predicted_change:.2f}")
            
            # 如果不是整点，等待到下一个整点
            if current_time.minute != 0:
                wait_until_next_hour()
                current_time = datetime.now()
                # 重新获取1小时数据
                df_1h = get_realtime_data(client, symbol, interval='1h')
                future_prices = predict_future(df_1h, modelnn, minmax, timestamp, future_hours=trend_window)
                predicted_change = future_prices[-1] - df_1h['close'].iloc[-1]
                last_prediction_time = current_time
                logging.info(f"整点新的预测完成: 预测变化 = {predicted_change:.2f}")

        # 获取最近的收盘价
        current_price = df_5m['close'].iloc[-1]
        
        # 计算RSI
        rsi_5m = calculate_rsi(df_5m, period=14)
        rsi_15m = calculate_rsi(df_15m, period=14)
        rsi_30m = calculate_rsi(df_30m, period=14)
        
        rsi_average = (rsi_5m.iloc[-1] + rsi_15m.iloc[-1] + rsi_30m.iloc[-1]) / 3
        
        # 动态阈值计算
        buy_threshold, sell_threshold = dynamic_threshold(predicted_change, 30, 70)
        
        # 记录详细的交易信息
        log_trade_info(current_time, current_price, predicted_change, rsi_5m, rsi_15m, rsi_30m, rsi_average, buy_threshold, sell_threshold)
        
        # 交易逻辑
        if (rsi_5m.iloc[-1] < buy_threshold and rsi_15m.iloc[-1] < buy_threshold and 
            rsi_30m.iloc[-1] < buy_threshold and rsi_average < buy_threshold):
            # 执行买入
            quantity = buy_amount / current_price
            order = execute_trade(client, symbol, "BUY", quantity)
            if order:
                logging.info(f"买入信号触发 - 价格: {current_price:.2f}, 数量: {quantity:.4f}")
                logging.info(f"买入订单详情: {order}")
            else:
                logging.error("买入订单执行失败")
        
        elif (rsi_5m.iloc[-1] > sell_threshold and rsi_15m.iloc[-1] > sell_threshold and 
              rsi_30m.iloc[-1] > sell_threshold and rsi_average > sell_threshold):
            # 执行卖出
            quantity = max_sell
            order = execute_trade(client, symbol, "SELL", quantity)
            if order:
                logging.info(f"卖出信号触发 - 价格: {current_price:.2f}, 数量: {quantity:.4f}")
                logging.info(f"卖出订单详情: {order}")
            else:
                logging.error("卖出订单执行失败")
        
        else:
            logging.info("当前不满足交易条件，继续观察")
            
async def start(update: Update, context: CallbackContext) -> None:
    """处理 /start 命令"""
    await update.message.reply_text('欢迎使用交易机器人！')

async def account(update: Update, context: CallbackContext) -> None:
    """处理 /account 命令"""
    account_info = client.get_account()
    balances = account_info['balances']
    message = "账户余额:\n"
    for balance in balances:
        asset = balance['asset']
        free = balance['free']
        locked = balance['locked']
        message += f"{asset}: 可用余额 {free}, 冻结余额 {locked}\n"
    await update.message.reply_text(message)

async def open_orders(update: Update, context: CallbackContext) -> None:
    """处理 /open_orders 命令"""
    orders = client.get_open_orders()
    if orders:
        message = "当前未完成订单:\n"
        for order in orders:
            message += f"订单ID: {order['orderId']}, 交易对: {order['symbol']}, 数量: {order['origQty']}, 价格: {order['price']}, 状态: {order['status']}\n"
    else:
        message = "没有未完成的订单。"
    await update.message.reply_text(message)

async def order_history(update: Update, context: CallbackContext) -> None:
    """处理 /order_history 命令"""
    orders = view_order_history(client, "BTCUSDT")
    if orders:
        message = "订单历史记录:\n"
        for order in orders:
            message += f"订单ID: {order['orderId']}, 交易对: {order['symbol']}, 价格: {order['price']}, 数量: {order['origQty']}, 状态: {order['status']}, 类型: {order['type']}, 时间: {order['time']}\n"
    else:
        message = "没有订单历史记录。"
    await update.message.reply_text(message)

async def latest_prediction(update: Update, context: CallbackContext) -> None:
    """处理 /latest_prediction 命令"""
    if latest_prediction_info:
        message = (
            f"最新预测信息:\n"
            f"当前时间: {latest_prediction_info['current_time']}\n"
            f"当前价格: {latest_prediction_info['current_price']:.2f}\n"
            f"预测价格变化: {latest_prediction_info['predicted_change']:.2f}\n"
            f"RSI (5分钟): {latest_prediction_info['rsi_5m'].iloc[-1]:.2f}\n"
            f"RSI (15分钟): {latest_prediction_info['rsi_15m'].iloc[-1]:.2f}\n"
            f"RSI (30分钟): {latest_prediction_info['rsi_30m'].iloc[-1]:.2f}\n"
            f"RSI 平均值: {latest_prediction_info['rsi_average']:.2f}\n"
            f"买入阈值: {latest_prediction_info['buy_threshold']:.2f}\n"
            f"卖出阈值: {latest_prediction_info['sell_threshold']:.2f}"
        )
    else:
        message = "没有最新的预测信息。"
    await update.message.reply_text(message)

def calculate_rsi(data, period=14):
    """计算RSI"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def normalize(value, min_value, max_value):
    """归一化函数"""
    return 2 * (value - min_value) / (max_value - min_value) - 1

def dynamic_threshold(predicted_change, base_buy_threshold, base_sell_threshold):
    """动态调整RSI阈值"""
    normalized_predicted_change = normalize(predicted_change, -2000, 2000)
    adjustment_factor = 50
    weight = 0.6
    if normalized_predicted_change > 0:
        buy_threshold = base_buy_threshold + adjustment_factor * normalized_predicted_change
        sell_threshold = base_sell_threshold + adjustment_factor * normalized_predicted_change * weight
        logging.info(f"买入阈值: {buy_threshold:.2f},模型加权：{normalized_predicted_change*adjustment_factor:.2f}")
        logging.info(f"卖出阈值: {sell_threshold:.2f},模型加权：{normalized_predicted_change*adjustment_factor*weight:.2f}")
    else:
        buy_threshold = base_buy_threshold + adjustment_factor * normalized_predicted_change * weight
        sell_threshold = base_sell_threshold + adjustment_factor * normalized_predicted_change
        logging.info(f"买入阈值: {buy_threshold:.2f},模型加权：{normalized_predicted_change*adjustment_factor*weight:.2f}")
        logging.info(f"卖出阈值: {sell_threshold:.2f},模型加权：{normalized_predicted_change*adjustment_factor:.2f}")
    return buy_threshold, sell_threshold
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext
def main():
    """启动Telegram机器人"""
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("account", account))
    application.add_handler(CommandHandler("open_orders", open_orders))
    application.add_handler(CommandHandler("order_history", order_history))
    application.add_handler(CommandHandler("latest_prediction", latest_prediction))  # 添加新的命令处理程序

    application.run_polling()
if __name__ == '__main__':
    # 创建并启动一个线程来运行交易策略
    strategy_thread = threading.Thread(target=run_strategy)
    strategy_thread.start()

    main()