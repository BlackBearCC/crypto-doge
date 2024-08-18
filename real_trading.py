import time

from binance.client import Client

api_key = "HdznWOVO7qLVgG3448j8s5ERAuH98a4GalNJEAMkmGfDrJgPkDKrWx39K7gCA1HU"
api_secret = "RfOQ6kh0DrvniBNPjrEBwTn0F3PLwSuWQnRgKtGsESs5xnCVKAIUYxthUAUTd7zJ"
client = Client(api_key, api_secret, testnet=True)


timestamp = int(time.time() * 1000)  # 以毫秒为单位

account_info = client.get_account(timestamp=timestamp)

# 打印账户信息
print(account_info)

symbol = "BTCUSDT"  # 合约对
side = "BUY"  # 买（BUY）或卖（SELL）
quantity = 1  # 数量
order_type = "MARKET"  # 市价单
position_side = "BOTH"  # 长空皆可（根据实际需求选择）

try:
    order = client.futures_create_order(
        symbol=symbol,
        side=side,
        type=order_type,
        quantity=quantity,
        positionSide=position_side,
    )
    print(order)
except Exception as e:
    print(f"下单失败: {str(e)}")