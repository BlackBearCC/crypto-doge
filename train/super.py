import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from tabulate import tabulate
from fpdf import FPDF
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

    # 数据清洗 - 去除插针现象
    combined_df['price_change'] = combined_df['close'].pct_change()
    combined_df = combined_df[(combined_df['price_change'] > -0.2) & (combined_df['price_change'] < 0.2)]
    combined_df.drop(columns=['price_change'], inplace=True)

    # 数据平滑 - 移动平均线
    combined_df['SMA_3'] = combined_df['close'].rolling(window=3).mean()
    combined_df['SMA_7'] = combined_df['close'].rolling(window=7).mean()

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
    for col in ['close', 'RSI', 'ATR', 'MACD', 'BB_High', 'BB_Low', 'SMA_3', 'SMA_7']:
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
features = ['RSI', 'ATR', 'MACD', 'BB_High', 'BB_Low', 'SMA_3', 'SMA_7', 'Lag_close', 'Lag_RSI', 'Lag_ATR', 'Lag_MACD', 'Lag_BB_High', 'Lag_BB_Low', 'Lag_SMA_3', 'Lag_SMA_7', 'Hour', 'DayOfWeek']
X = data[features]
y = data['Price_Change']
direction = data['Direction']

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

# 无监督学习特征提取
# 聚类
kmeans = KMeans(n_clusters=5, n_init=20, random_state=42)  # 显式设置 n_init
X_train_cluster = kmeans.fit_predict(X_train)
X_test_cluster = kmeans.predict(X_test)

# PCA降维
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 将聚类标签和PCA特征添加到原始特征中
X_train = np.hstack((X_train, X_train_cluster.reshape(-1, 1), X_train_pca))
X_test = np.hstack((X_test, X_test_cluster.reshape(-1, 1), X_test_pca))

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 特征选择
lasso = Lasso(alpha=0.1, max_iter=10000)  # 增加alpha和最大迭代次数
lasso.fit(X_train, y_train)
model = SelectFromModel(lasso, prefit=True)
X_train_selected = model.transform(X_train)
X_test_selected = model.transform(X_test)

# 训练基础模型
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_jobs=-1),
    'Support Vector Regressor': SVR(),
    'LightGBM': lgb.LGBMRegressor(device='gpu', n_jobs=-1),
    'XGBoost': XGBRegressor(tree_method="hist", device="cuda", n_jobs=-1)
}

trained_models = {}
for name, model in tqdm(models.items(), desc="Training models"):
    logger.info("Training %s model", name)
    if name in ['LightGBM', 'XGBoost']:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }

        halving_search = HalvingGridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        halving_search.fit(X_train_selected, y_train)
        best_model = halving_search.best_estimator_
        trained_models[name] = best_model
        logger.info("%s model training completed with best parameters: %s", name, halving_search.best_params_)
    elif name == 'Random Forest Regressor':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'max_features': ['sqrt', 'log2', None]
        }

        halving_search = HalvingGridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        halving_search.fit(X_train_selected, y_train)
        best_model = halving_search.best_estimator_
        trained_models[name] = best_model
        logger.info("%s model training completed with best parameters: %s", name, halving_search.best_params_)
    else:
        model.fit(X_train_selected, y_train)
        trained_models[name] = model
        logger.info("%s model training completed", name)

# 使用基础模型的预测准确率作为新特征
X_train_meta = pd.DataFrame(X_train_selected).copy()
X_test_meta = pd.DataFrame(X_test_selected).copy()

# 将所有列名转换为字符串类型
X_train_meta.columns = X_train_meta.columns.astype(str)
X_test_meta.columns = X_test_meta.columns.astype(str)

for name, model in trained_models.items():
    logger.info(f"Generating predictions for {name} model")
    X_train_meta[f'{name}_pred'] = model.predict(X_train_selected)
    X_test_meta[f'{name}_pred'] = model.predict(X_test_selected)

# 数据标准化
scaler_meta = StandardScaler()
X_train_meta = scaler_meta.fit_transform(X_train_meta)
X_test_meta = scaler_meta.transform(X_test_meta)

# 使用新的特征进行再训练
meta_model = RandomForestRegressor(random_state=42, n_jobs=-1)
meta_model.fit(X_train_meta, y_train)
logger.info("Meta model training completed")

# 评估模型函数
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

    results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Accuracy': accuracy
    }

    return y_pred, results

# 评估所有模型并保存结果
results_dict = {}
for name, model in trained_models.items():
    logger.info(f"Evaluating {name} Model:")
    _, results = evaluate_model(model, X_test_selected, y_test, y_test_direction)
    results_dict[name] = results

logger.info("Evaluating Meta Model:")
_, results = evaluate_model(meta_model, X_test_meta, y_test, y_test_direction)
results_dict['Meta Model'] = results

# 将评估结果保存为DataFrame
results_df = pd.DataFrame(results_dict).T

# 使用tabulate生成漂亮的表格
table = tabulate(results_df, headers='keys', tablefmt='pipe')

# 打印表格到控制台
print(table)

# 将表格保存为PDF文件
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Model Evaluation Results', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()
pdf.chapter_title('Model Evaluation Results')
pdf.chapter_body(table)
pdf.output('model_evaluation_results.pdf')

# 显示评估结果表
print(results_df)
