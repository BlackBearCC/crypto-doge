from datetime import datetime
import joblib
import matplotlib.pyplot as plt
# 必须在使用 HalvingGridSearchCV 之前导入
from sklearn.experimental import enable_halving_search_cv  # 必须显式导入
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from tabulate import tabulate
from fpdf import FPDF
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 读取CSV文件并预处理数据
def read_and_prepare_data(filepath):
    logger.info("Reading and preparing data from file: %s", filepath)
    df = pd.read_parquet(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('timestamp', inplace=True)

    # 数据清洗 - 去除极端的价格波动
    df['price_change'] = df['close'].pct_change()
    df = df[(df['price_change'] > -0.2) & (df['price_change'] < 0.2)]
    df.drop(columns=['price_change'], inplace=True)

    # 数据平滑 - 移动平均线
    df['SMA_3'] = df['close'].rolling(window=3).mean()
    df['SMA_7'] = df['close'].rolling(window=7).mean()

    # 计算技术指标
    logger.info("Calculating technical indicators")
    tqdm.pandas(desc="Calculating technical indicators")
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
    df['Direction'] = (df['Price_Change'] > 0).astype(int)  # 1表示涨，0表示跌

    # 确认标签数据的分布
    print(df['Price_Change'].describe())
    print(df['Direction'].value_counts())

    # 删除缺失值
    df.dropna(inplace=True)
    logger.info("Data preparation completed")
    return df

# 读取并处理数据
data = read_and_prepare_data('../BTC_USDT_ohlcv_data.parquet')

# 准备特征和标签
features = ['RSI', 'ATR', 'MACD', 'BB_High', 'BB_Low', 'SMA_3', 'SMA_7',
            'Lag_close', 'Lag_RSI', 'Lag_ATR', 'Lag_MACD', 'Lag_BB_High',
            'Lag_BB_Low', 'Lag_SMA_3', 'Lag_SMA_7', 'Hour', 'DayOfWeek']
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

# 再次标准化添加特征后的数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 特征选择
lasso = Lasso(alpha=0.1, max_iter=10000)  # 增加alpha和最大迭代次数
lasso.fit(X_train, y_train)
model = SelectFromModel(lasso, prefit=True)
X_train_selected = model.transform(X_train)
X_test_selected = model.transform(X_test)

# 获取被选择的特征掩码（布尔数组，True 表示被选择的特征）
selected_features_mask = model.get_support()

# 获取选择的特征名称
final_features = [feature for feature, selected in zip(features + ['cluster', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5'], selected_features_mask) if selected]
print("Selected Features:", final_features)

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

    # 保存基础模型
    joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')
    logger.info("%s model saved successfully", name)

# -----------------------------------------------堆叠集成模型---------------------------------------------------------
# 使用基础模型的预测作为新特征
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

# 使用新的特征进行堆叠集成再训练
stacking_model = StackingRegressor(
    estimators=[(name, model) for name, model in trained_models.items()],
    final_estimator=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
)
stacking_model.fit(X_train_meta, y_train)
logger.info("Stacking model training completed")

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

# 保存模型
def save_model(model, model_path):
    logger.info(f"Saving {model.__class__.__name__} model...")
    joblib.dump(model, model_path)
    # 检查文件是否存在
    if Path(model_path).exists():
        logger.info(f"Model successfully saved to {model_path}")
    else:
        logger.error(f"Failed to save the model to {model_path}")

# 保存堆叠集成模型
save_model(stacking_model, '../stacking_model.pkl')

# 评估所有模型并保存结果
results_dict = {}
for name, model in trained_models.items():
    logger.info(f"Evaluating {name} Model:")
    _, results = evaluate_model(model, X_test_selected, y_test, y_test_direction)
    results_dict[name] = results

logger.info("Evaluating Stacking Model:")
_, results = evaluate_model(stacking_model, X_test_meta, y_test, y_test_direction)
results_dict['Stacking Model'] = results

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

# 获取当前日期和时间
now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
pdf.output(f'{date_time_str}_model_evaluation_results.pdf')

# 显示评估结果表
print(results_df)
