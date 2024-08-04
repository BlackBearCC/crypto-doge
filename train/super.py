import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import SMAIndicator, EMAIndicator, MACD
import lightgbm as lgb
from xgboost import XGBRegressor
# import shap
from tqdm import tqdm

# 读取CSV文件并预处理数据
def read_and_prepare_data(directory):
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
    tqdm.pandas(desc="Calculating technical indicators")
    combined_df['RSI'] = RSIIndicator(combined_df['close']).rsi()
    combined_df['ATR'] = AverageTrueRange(combined_df['high'], combined_df['low'], combined_df['close']).average_true_range()
    combined_df['SMA'] = SMAIndicator(combined_df['close'], window=100).sma_indicator()
    combined_df['EMA'] = EMAIndicator(combined_df['close'], window=50).ema_indicator()
    combined_df['MACD'] = MACD(combined_df['close']).macd()
    combined_df['Stochastic'] = StochasticOscillator(combined_df['high'], combined_df['low'], combined_df['close']).stoch()
    bb = BollingerBands(combined_df['close'])
    combined_df['Bollinger_High'] = bb.bollinger_hband()
    combined_df['Bollinger_Low'] = bb.bollinger_lband()

    # 创建标签：未来15分钟的价格变化
    combined_df['Price_Change'] = combined_df['close'].shift(-1) - combined_df['close']
    combined_df['Direction'] = (combined_df['Price_Change'] > 0).astype(int)  # 1表示涨，0表示跌

    # 删除缺失值
    combined_df.dropna(inplace=True)

    return combined_df

data = read_and_prepare_data('../BTCUSDT-1h')

# 准备特征和标签
X = data[['RSI', 'ATR', 'SMA', 'EMA', 'MACD', 'Stochastic', 'Bollinger_High', 'Bollinger_Low']]
y = data['Price_Change']

# 分割数据集
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_direction = (y_train > 0).astype(int)
y_test_direction = (y_test > 0).astype(int)

# 训练模型
models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(),
    'LightGBM': lgb.LGBMRegressor(device='gpu'),
    'XGBoost': XGBRegressor(tree_method='gpu_hist')
}

trained_models = {}
for name, model in tqdm(models.items(), desc="Training models"):
    model.fit(X_train, y_train)
    trained_models[name] = model

# 7. 集成模型
print("Training Ensemble Model...")
ensemble_model = VotingRegressor(estimators=[
    ('lr', trained_models['Linear Regression']),
    ('gbr', trained_models['Gradient Boosting Regressor']),
    ('rf', trained_models['Random Forest Regressor']),
    ('svr', trained_models['Support Vector Regressor']),
    ('lgb', trained_models['LightGBM']),
    ('xgb', trained_models['XGBoost'])
])
ensemble_model.fit(X_train, y_train)


# 评估模型
def evaluate_model(model, X_test, y_test, y_test_direction):
    y_pred = model.predict(X_test)
    y_pred_direction = (y_pred > 0).astype(int)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = (y_pred_direction == y_test_direction).mean()

    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Prediction Accuracy: {accuracy:.2%}')

    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Price Change')
    plt.ylabel('Predicted Price Change')
    plt.title('Actual vs Predicted Price Change')
    plt.show()

    return y_pred


for name, model in trained_models.items():
    print(f"Evaluating {name} Model:")
    evaluate_model(model, X_test, y_test, y_test_direction)

print("Evaluating Ensemble Model:")
y_pred_ensemble = evaluate_model(ensemble_model, X_test, y_test, y_test_direction)

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

print("Random Forest Feature Importance:")
plot_feature_importance(trained_models['Random Forest Regressor'], X.columns)

print("Gradient Boosting Regressor Feature Importance:")
plot_feature_importance(trained_models['Gradient Boosting Regressor'], X.columns)

print("XGBoost Feature Importance:")
plot_feature_importance(trained_models['XGBoost'], X.columns)

print("LightGBM Feature Importance:")
plot_feature_importance(trained_models['LightGBM'], X.columns)

# 保存模型
print("Saving model...")
joblib.dump(ensemble_model, '../ensemble_model.pkl')
print("Model saved.")
