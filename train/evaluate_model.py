import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, learning_curve, validation_curve, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import MACD
from fpdf import FPDF
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 交叉验证
def cross_validate_model(model, X, y, cv=5):
    tscv = TimeSeriesSplit(n_splits=cv)
    cv_results = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    logger.info(f'Cross-Validation MSE: {cv_results.mean()} ± {cv_results.std()}')
    return cv_results

# 学习曲线
def plot_learning_curve(model, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.title('Learning Curve')
    plt.show()

# 验证曲线
def plot_validation_curve(model, X, y, param_name, param_range, cv=5):
    train_scores, test_scores = validation_curve(model, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(param_range, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.xlabel(param_name)
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.title(f'Validation Curve ({param_name})')
    plt.show()

# 特征重要性分析
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()

# 将评估结果保存为PDF文件
def save_results_as_pdf(table):
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

# 从文件夹读取并预处理数据
def read_and_prepare_data(directory):
    logger.info("Reading and preparing data from directory: %s", directory)
    data_folder = Path(directory)
    all_files = list(data_folder.glob('*.csv'))
    df_list = []

    for file_path in all_files:
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

# 评估元模型
def evaluate_meta_model(meta_model_path, X, y):
    meta_model = joblib.load(meta_model_path)

    # 交叉验证评估
    cross_validate_model(meta_model, X, y)

    # 绘制学习曲线
    plot_learning_curve(meta_model, X, y)

    # 验证曲线（以Random Forest为例，调节n_estimators参数）
    plot_validation_curve(meta_model, X, y, param_name='n_estimators', param_range=[50, 100, 200, 300])

    # 特征重要性分析
    feature_names_meta = list(X.columns)
    plot_feature_importance(meta_model, feature_names_meta)

    # 最终验证集评估
    X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(X, y, test_size=0.2, random_state=42)
    meta_model.fit(X_final_train, y_final_train)
    y_final_pred = meta_model.predict(X_final_val)
    final_mse = mean_squared_error(y_final_val, y_final_pred)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(y_final_val, y_final_pred)
    final_accuracy = (y_final_pred > 0).astype(int).mean()

    logger.info(f'Final Validation MSE: {final_mse}')
    logger.info(f'Final Validation RMSE: {final_rmse}')
    logger.info(f'Final Validation MAE: {final_mae}')
    logger.info(f'Final Validation Accuracy: {final_accuracy:.2%}')

    # 打印结果表格
    results = {
        'MSE': final_mse,
        'RMSE': final_rmse,
        'MAE': final_mae,
        'Accuracy': final_accuracy
    }
    results_df = pd.DataFrame(results, index=[0])
    from tabulate import tabulate
    table = tabulate(results_df, headers='keys', tablefmt='pipe')
    print(table)

    # 保存结果为PDF
    save_results_as_pdf(table)

if __name__ == "__main__":
    # 加载并预处理数据
    data = read_and_prepare_data('../BTCUSDT-1h')
    features = ['RSI', 'ATR', 'MACD', 'BB_High', 'BB_Low', 'SMA_3', 'SMA_7', 'Lag_close', 'Lag_RSI', 'Lag_ATR', 'Lag_MACD', 'Lag_BB_High', 'Lag_BB_Low', 'Lag_SMA_3', 'Lag_SMA_7', 'Hour', 'DayOfWeek']
    X = data[features]
    y = data['Price_Change']

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 评估元模型
    evaluate_meta_model('../meta_model.pkl', pd.DataFrame(X, columns=features), y)
