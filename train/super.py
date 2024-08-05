import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import MACD
import lightgbm as lgb
from xgboost import XGBRegressor
from tqdm import tqdm
import mplfinance as mpf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 读取CSV文件并预处理数据
def read_and_prepare_data(directory):
    logger.info("Reading and preparing data from directory: %s", directory)
    data_folder = Path(directory)
    all_files = list(data_folder.glob('*.csv'))
    df_list = []

    for file_path in tqdm(all_files, desc="Reading CSV files"):
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)
        df_list.append(df)

    combined_df = pd.concat(df_list)
    combined_df.sort_index(inplace=True)

    # 计算技术指标
    logger.info("Calculating technical indicators")
    tqdm.pandas(desc="Calculating technical indicators")
    combined_df['RSI'] = RSIIndicator(combined_df['close']).rsi()
    combined_df['ATR'] = AverageTrueRange(combined_df['high'], combined_df['low'], combined_df['close']).average_true_range()
    combined_df['MACD'] = MACD(combined_df['close']).macd()
    bb_indicator = BollingerBands(combined_df['close'])
    combined_df['BB_High'] = bb_indicator.bollinger_hband()
    combined_df['BB_Low'] = bb_indicator.bollinger_lband()

    # 创建滞后特征
    for col in ['close', 'RSI', 'ATR', 'MACD', 'BB_High', 'BB_Low']:
        combined_df[f'Lag_{col}'] = combined_df[col].shift(1)

    # 创建时间特征
    combined_df['Hour'] = combined_df.index.hour
    combined_df['DayOfWeek'] = combined_df.index.dayofweek

    # 创建标签：未来15分钟的价格变化
    combined_df['Price_Change'] = combined_df['close'].shift(-1) - combined_df['close']
    combined_df['Direction'] = (combined_df['Price_Change'] > 0).astype(int)  # 1表示涨，0表示跌

    # 删除缺失值
    combined_df.dropna(inplace=True)
    logger.info("Data preparation completed")

    return combined_df

data = read_and_prepare_data('../BTCUSDT-1h')

# 准备特征和标签
features = ['RSI', 'ATR', 'MACD', 'BB_High', 'BB_Low', 'Lag_close', 'Lag_RSI', 'Lag_ATR', 'Lag_MACD', 'Lag_BB_High', 'Lag_BB_Low', 'Hour', 'DayOfWeek']
X = data[features]
y = data['Price_Change']

# 数据平衡
logger.info("Balancing the dataset")
# 分离多数类和少数类
majority_class = data[data['Direction'] == 0]
minority_class = data[data['Direction'] == 1]

# 过采样少数类
upsampled_minority_class = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
data_balanced = pd.concat([majority_class, upsampled_minority_class])

# 准备特征和标签
X_balanced = data_balanced[features]
y_balanced = data_balanced['Price_Change']

# 数据标准化
scaler = StandardScaler()
X_balanced = scaler.fit_transform(X_balanced)

# 分割数据集
logger.info("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

y_train_direction = (y_train > 0).astype(int)
y_test_direction = (y_test > 0).astype(int)

# 训练模型
models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'LightGBM': lgb.LGBMRegressor(device='gpu'),
    'XGBoost': XGBRegressor(tree_method='gpu_hist')
}

trained_models = {}
for name, model in tqdm(models.items(), desc="Training models"):
    logger.info("Training %s model", name)
    if name in ['Gradient Boosting Regressor', 'LightGBM', 'XGBoost']:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        trained_models[name] = best_model
        logger.info("%s model training completed with best parameters: %s", name, grid_search.best_params_)
    elif name == 'Random Forest Regressor':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        trained_models[name] = best_model
        logger.info("%s model training completed with best parameters: %s", name, grid_search.best_params_)
    else:
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info("%s model training completed", name)

# 集成模型
logger.info("Training Ensemble Model...")
ensemble_model = VotingRegressor(estimators=[
    ('lr', trained_models['Linear Regression']),
    ('gbr', trained_models['Gradient Boosting Regressor']),
    ('rf', trained_models['Random Forest Regressor']),
    ('svr', trained_models['Support Vector Regressor']),
    ('lgb', trained_models['LightGBM']),
    ('xgb', trained_models['XGBoost'])
])
ensemble_model.fit(X_train, y_train)
logger.info("Ensemble model training completed")


# LSTM 模型
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 准备LSTM数据
def prepare_lstm_data(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 10
X_train_lstm, y_train_lstm = prepare_lstm_data(pd.DataFrame(X_train, columns=features), y_train, time_steps)
X_test_lstm, y_test_lstm = prepare_lstm_data(pd.DataFrame(X_test, columns=features), y_test, time_steps)

# 训练LSTM模型
logger.info("Training LSTM model...")
lstm_model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),
    ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss', mode='min')
]
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, validation_split=0.2, callbacks=callbacks)
logger.info("LSTM model training completed")


# 评估LSTM模型
def evaluate_lstm_model(model, X_test, y_test, y_test_direction):
    logger.info("Evaluating LSTM model...")
    y_pred = model.predict(X_test)
    y_pred_direction = (y_pred.flatten() > 0).astype(int)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = (y_pred_direction == y_test_direction[:len(y_pred_direction)]).mean()

    logger.info(f'Mean Squared Error: {mse}')
    logger.info(f'Root Mean Squared Error: {rmse}')
    logger.info(f'Mean Absolute Error: {mae}')
    logger.info(f'Prediction Accuracy: {accuracy:.2%}')

    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Price Change')
    plt.ylabel('Predicted Price Change')
    plt.title('Actual vs Predicted Price Change')
    plt.show()

    return y_pred


logger.info("Evaluating LSTM Model:")
evaluate_lstm_model(lstm_model, X_test_lstm, y_test_lstm, y_test_direction[time_steps:])


# 评估其他模型
def evaluate_model(model, X_test, y_test, y_test_direction):
    logger.info(f"Evaluating {model.__class__.__name__} model...")
    y_pred = model.predict(X_test)
    y_pred_direction = (y_pred > 0).astype(int)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = (y_pred_direction == y_test_direction).mean()

    logger.info(f'Mean Squared Error: {mse}')
    logger.info(f'Root Mean Squared Error: {rmse}')
    logger.info(f'Mean Absolute Error: {mae}')
    logger.info(f'Prediction Accuracy: {accuracy:.2%}')

    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Price Change')
    plt.ylabel('Predicted Price Change')
    plt.title('Actual vs Predicted Price Change')
    plt.show()

    return y_pred


# 获取预测价格
def get_predicted_prices(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return y_test.index, y_test.values, y_pred


for name, model in trained_models.items():
    logger.info(f"Evaluating {name} Model:")
    evaluate_model(model, X_test, y_test, y_test_direction)

logger.info("Evaluating Ensemble Model:")
y_pred_ensemble = evaluate_model(ensemble_model, X_test, y_test, y_test_direction)


# 绘制K线图
def plot_kline(data, actual_prices, predicted_prices, title='Actual vs Predicted Prices'):
    data['Actual'] = actual_prices
    data['Predicted'] = predicted_prices
    apd = [mpf.make_addplot(data['Actual'], color='blue'),
           mpf.make_addplot(data['Predicted'], color='red')]
    mpf.plot(data, type='candle', addplot=apd, style='charles', title=title)


# 获取实际和预测价格
actual_index, actual_prices, predicted_prices = get_predicted_prices(ensemble_model, X_test, y_test)
data_for_plot = data.loc[actual_index]

# 绘制实际价格和预测价格K线图
plot_kline(data_for_plot, actual_prices, predicted_prices)


# 获取特征重要性
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()


logger.info("Random Forest Feature Importance:")
plot_feature_importance(trained_models['Random Forest Regressor'], features)

logger.info("Gradient Boosting Regressor Feature Importance:")
plot_feature_importance(trained_models['Gradient Boosting Regressor'], features)

logger.info("XGBoost Feature Importance:")
plot_feature_importance(trained_models['XGBoost'], features)

logger.info("LightGBM Feature Importance:")
plot_feature_importance(trained_models['LightGBM'], features)

# 保存模型
logger.info("Saving model...")
joblib.dump(ensemble_model, '../54%_ensemble_model.pkl')
logger.info("Model saved.")
