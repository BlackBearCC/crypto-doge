import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import MACD


# 加载训练数据集
def load_data(filepath):
    df = pd.read_parquet(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('timestamp', inplace=True)

    # 数据清洗
    df['price_change'] = df['close'].pct_change()
    df = df[(df['price_change'] > -0.2) & (df['price_change'] < 0.2)]
    df.drop(columns=['price_change'], inplace=True)

    # 数据平滑
    df['SMA_3'] = df['close'].rolling(window=3).mean()
    df['SMA_7'] = df['close'].rolling(window=7).mean()

    # 计算技术指标
    df['RSI'] = RSIIndicator(df['close']).rsi()
    df['ATR'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['MACD'] = MACD(df['close']).macd()
    bb_indicator = BollingerBands(df['close'])
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low'] = bb_indicator.bollinger_lband()

    # 创建滞后特征
    for col in ['close', 'RSI', 'ATR', 'MACD', 'BB_High', 'BB_Low', 'SMA_3', 'SMA_7']:
        df[f'Lag_{col}'] = df[col].shift(1)

    # 创建时间特征
    df['Minute'] = df.index.minute
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek

    # 创建标签：未来15分钟的价格变化
    df['Price_Change'] = df['close'].shift(-1) - df['close']
    df.dropna(inplace=True)

    return df


# 加载模型
def load_model(model_path):
    model = joblib.load(model_path)
    return model


# 评估模型
def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R^2 Score: {r2}')

    # 打印预测结果和实际结果的对比
    comparison_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
    print(comparison_df.head(2000))  # 打印前20条数据
    comparison_df.to_csv('model_evaluation_results.csv')
    return comparison_df


if __name__ == "__main__":
    # 文件路径
    data_filepath = '../BTC_USDT_ohlcv_data.parquet'
    model_filepath = '../meta_model.pkl'

    # 加载数据和模型
    data = load_data(data_filepath)
    model = load_model(model_filepath)

    # 提取特征和标签（确保只选择11个特征）
    features = ['RSI', 'ATR', 'MACD', 'BB_High', 'BB_Low',
                'SMA_3', 'SMA_7', 'Lag_close', 'Lag_RSI',
                'Lag_ATR', 'Lag_MACD']
    X = data[features]
    y = data['Price_Change']

    # 重新标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 评估模型
    evaluate_model(model, X, y)
