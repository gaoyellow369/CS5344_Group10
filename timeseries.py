#不要原版的lightgbm和xgboost

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from aeon.regression.interval_based import TimeSeriesForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# 数据预处理函数
def preprocess_data(input_file='input_attributes.csv', labels_file='labels.csv', num_lags=3):
    input_data = pd.read_csv(input_file, encoding='GBK')
    labels_data = pd.read_csv(labels_file, encoding='GBK', index_col=0)

    input_data = input_data.iloc[::-1].reset_index(drop=True)
    last_column = input_data.iloc[:, -1]
    input_data = input_data.iloc[:, :-1]

    days = input_data.shape[0] // 24
    input_matrices = []
    labels_matrices = []

    labels_data = labels_data.T
    labels_data.index = range(1, labels_data.shape[0] + 1)

    removed_days = []
    for day in range(days):
        daily_data = input_data.iloc[day * 24:(day + 1) * 24, 1:25].values
        last_column_day = last_column.iloc[day * 24:(day + 1) * 24]

        if np.isnan(daily_data).any() or labels_data.iloc[day].isnull().any() or last_column_day.notna().any():
            removed_days.append(day + 1)
            continue
        input_matrices.append(daily_data)
        labels_matrices.append(labels_data.iloc[day].values)

    if removed_days:
        print(f"剔除的天数：{removed_days}")
    else:
        print("没有剔除任何天数的数据")

    remaining_days = len(input_matrices)
    print(f"剩余的天数：{remaining_days}")

    X = np.array(input_matrices)
    y = np.array(labels_matrices)

    X_with_lags = []
    for i in range(num_lags, X.shape[0]):
        lagged_features = [X[i - lag] for lag in range(1, num_lags + 1)]
        combined_features = np.hstack([X[i]] + lagged_features)
        X_with_lags.append(combined_features)

    X_with_lags = np.array(X_with_lags)
    y = y[num_lags:]

    return X_with_lags, y

# 列归一化函数
def columnwise_normalization_fit(data):
    scalers = []
    for i in range(data.shape[2]):
        scaler = StandardScaler()
        scalers.append(scaler.fit(data[:, :, i]))
        data[:, :, i] = scalers[i].transform(data[:, :, i])
    return data, scalers

def columnwise_normalization_transform(data, scalers):
    for i in range(data.shape[2]):
        data[:, :, i] = scalers[i].transform(data[:, :, i])
    return data

# 计算评估指标函数
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    epsilon = 1e-10
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    mape = np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100
    return mse, mae, rmse, r2, mape

# 训练和堆叠模型函数
def train_model(X_train, X_test, y_train, y_test, label_columns=None, exclude_columns=None, column_names=None):
    if exclude_columns is not None and column_names is not None:
        excluded_column_names = [column_names[i] for i in exclude_columns]
        print(f"排除的列索引: {exclude_columns}")
        print(f"排除的列名: {excluded_column_names}")
        X_train = np.delete(X_train, exclude_columns, axis=2)
        X_test = np.delete(X_test, exclude_columns, axis=2)

    if label_columns is not None:
        y_train = y_train[:, label_columns]
        y_test = y_test[:, label_columns]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    train_results = []
    test_results = []
    stacked_train_predictions = []
    stacked_test_predictions = []
    stacked_metrics_train = []
    stacked_metrics_test = []

    for i in range(y_train.shape[1]):
        # 基于时间序列特征的 LightGBM
        lgb_model_ts = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        X_train_ts = X_train.reshape(X_train.shape[0], -1)
        X_test_ts = X_test.reshape(X_test.shape[0], -1)
        lgb_model_ts.fit(X_train_ts, y_train[:, i])
        y_train_pred_lgb_ts = lgb_model_ts.predict(X_train_ts)
        y_test_pred_lgb_ts = lgb_model_ts.predict(X_test_ts)

        # 基于时间序列特征的 XGBoost
        xgb_model_ts = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
        xgb_model_ts.fit(X_train_ts, y_train[:, i])
        y_train_pred_xgb_ts = xgb_model_ts.predict(X_train_ts)
        y_test_pred_xgb_ts = xgb_model_ts.predict(X_test_ts)

        # TimeSeriesForest
        ts_model = TimeSeriesForestRegressor(n_estimators=10, random_state=42)
        ts_model.fit(X_train, y_train[:, i])
        y_train_pred_ts = ts_model.predict(X_train)
        y_test_pred_ts = ts_model.predict(X_test)

        # 堆叠模型输入
        train_stacked_input = np.vstack((y_train_pred_lgb_ts, y_train_pred_xgb_ts, y_train_pred_ts)).T
        test_stacked_input = np.vstack((y_test_pred_lgb_ts, y_test_pred_xgb_ts, y_test_pred_ts)).T

        # Ridge 回归
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(train_stacked_input, y_train[:, i])
        y_train_pred_ridge = ridge_model.predict(train_stacked_input)
        y_test_pred_ridge = ridge_model.predict(test_stacked_input)

        # GradientBoosting 回归
        gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbr_model.fit(train_stacked_input, y_train[:, i])
        y_train_pred_gbr = gbr_model.predict(train_stacked_input)
        y_test_pred_gbr = gbr_model.predict(test_stacked_input)

        # 计算各模型评估指标
        train_metrics_lgb_ts = calculate_metrics(y_train[:, i], y_train_pred_lgb_ts)
        test_metrics_lgb_ts = calculate_metrics(y_test[:, i], y_test_pred_lgb_ts)
        train_metrics_xgb_ts = calculate_metrics(y_train[:, i], y_train_pred_xgb_ts)
        test_metrics_xgb_ts = calculate_metrics(y_test[:, i], y_test_pred_xgb_ts)
        train_metrics_ts = calculate_metrics(y_train[:, i], y_train_pred_ts)
        test_metrics_ts = calculate_metrics(y_test[:, i], y_test_pred_ts)
        train_metrics_ridge = calculate_metrics(y_train[:, i], y_train_pred_ridge)
        test_metrics_ridge = calculate_metrics(y_test[:, i], y_test_pred_ridge)
        train_metrics_gbr = calculate_metrics(y_train[:, i], y_train_pred_gbr)
        test_metrics_gbr = calculate_metrics(y_test[:, i], y_test_pred_gbr)

        train_results.append((train_metrics_lgb_ts, train_metrics_xgb_ts, train_metrics_ts))
        test_results.append((test_metrics_lgb_ts, test_metrics_xgb_ts, test_metrics_ts))
        stacked_train_predictions.append((train_metrics_ridge, train_metrics_gbr))
        stacked_test_predictions.append((test_metrics_ridge, test_metrics_gbr))

    return train_results, test_results, stacked_train_predictions, stacked_test_predictions

# 评估模型函数
def evaluate_model(X_train, X_test, y_train, y_test, label_columns, exclude_columns=None, column_names=None):
    train_results, test_results, stacked_metrics_train, stacked_metrics_test = train_model(
        X_train, X_test, y_train, y_test, label_columns, exclude_columns, column_names
    )

    metrics = {
        'Label': [], 'Model': [], 'Train_MSE': [], 'Test_MSE': [], 'Train_MAE': [], 'Test_MAE': [],
        'Train_RMSE': [], 'Test_RMSE': [], 'Train_R2': [], 'Test_R2': [], 'Train_MAPE': [], 'Test_MAPE': []
    }

    models = ['TS LightGBM', 'TS XGBoost', 'TSForest', 'RidgeStacked', 'GBRStacked']

    for i, (train, test) in enumerate(zip(train_results, test_results)):
        for j, model in enumerate(models[:-2]):
            metrics['Label'].append(f'Label {i + 1}')
            metrics['Model'].append(model)
            metrics['Train_MSE'].append(train[j][0])
            metrics['Test_MSE'].append(test[j][0])
            metrics['Train_MAE'].append(train[j][1])
            metrics['Test_MAE'].append(test[j][1])
            metrics['Train_RMSE'].append(train[j][2])
            metrics['Test_RMSE'].append(test[j][2])
            metrics['Train_R2'].append(train[j][3])
            metrics['Test_R2'].append(test[j][3])
            metrics['Train_MAPE'].append(train[j][4])
            metrics['Test_MAPE'].append(test[j][4])

    for i, (train_metrics, test_metrics) in enumerate(zip(stacked_metrics_train, stacked_metrics_test)):
        train_metrics_ridge, train_metrics_gbr = train_metrics
        test_metrics_ridge, test_metrics_gbr = test_metrics

        metrics['Label'].append(f'Label {i + 1}')
        metrics['Model'].append('RidgeStacked')
        metrics['Train_MSE'].append(train_metrics_ridge[0])
        metrics['Test_MSE'].append(test_metrics_ridge[0])
        metrics['Train_MAE'].append(train_metrics_ridge[1])
        metrics['Test_MAE'].append(test_metrics_ridge[1])
        metrics['Train_RMSE'].append(train_metrics_ridge[2])
        metrics['Test_RMSE'].append(test_metrics_ridge[2])
        metrics['Train_R2'].append(train_metrics_ridge[3])
        metrics['Test_R2'].append(test_metrics_ridge[3])
        metrics['Train_MAPE'].append(train_metrics_ridge[4])
        metrics['Test_MAPE'].append(test_metrics_ridge[4])

        metrics['Label'].append(f'Label {i + 1}')
        metrics['Model'].append('GBRStacked')
        metrics['Train_MSE'].append(train_metrics_gbr[0])
        metrics['Test_MSE'].append(test_metrics_gbr[0])
        metrics['Train_MAE'].append(train_metrics_gbr[1])
        metrics['Test_MAE'].append(test_metrics_gbr[1])
        metrics['Train_RMSE'].append(train_metrics_gbr[2])
        metrics['Test_RMSE'].append(test_metrics_gbr[2])
        metrics['Train_R2'].append(train_metrics_gbr[3])
        metrics['Test_R2'].append(test_metrics_gbr[3])
        metrics['Train_MAPE'].append(train_metrics_gbr[4])
        metrics['Test_MAPE'].append(test_metrics_gbr[4])

    return pd.DataFrame(metrics)

# 调用预处理函数
input_file = 'input_attributes.csv'
labels_file = 'labels.csv'
X, y = preprocess_data(input_file, labels_file)

# 获取列名
column_names = pd.read_csv(input_file, encoding='GBK').columns[1:25]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 指定标签列
label_columns = [1, 2, 3, 4, 5]  # 根据需要修改为你想要的标签索引

# 评估模型
metrics_df = evaluate_model(X_train, X_test, y_train, y_test, label_columns, exclude_columns=[10, 12, 21, 23], column_names=column_names)

# 输出结果
print(metrics_df)


def plot_metrics(metrics_df):
    # 定义评估指标
    metrics = ['MSE', 'MAE', 'RMSE', 'R2', 'MAPE']
    data_types = ['Train', 'Test']

    # 确保模型顺序为 'TSForest' 最左，其余保持原顺序
    model_order = ['TSForest', 'TS LightGBM', 'TS XGBoost', 'RidgeStacked', 'GBRStacked']
    metrics_df['Model'] = pd.Categorical(metrics_df['Model'], categories=model_order, ordered=True)
    metrics_df = metrics_df.sort_values(by='Model')

    # 将 Model 列转换为字符串类型
    metrics_df['Model'] = metrics_df['Model'].astype(str)

    # 设置 Label 顺序
    label_order = sorted(metrics_df['Label'].unique(), key=lambda x: int(x.split(' ')[-1]))
    
    for label in label_order:  # 按照 Label 顺序进行绘图
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for data_type in data_types:
                # 过滤数据，获取当前标签和数据类型的指标
                filtered_data = metrics_df[metrics_df['Label'] == label]
                plt.plot(
                    filtered_data['Model'], 
                    filtered_data[f'{data_type}_{metric}'], 
                    marker='o', 
                    label=f'{data_type} {metric}'
                )
            
            # 设置图表标题和标签
            plt.title(f'{metric} for {label} with Various Models')
            plt.xlabel('Model')
            plt.ylabel(metric)
            plt.legend()
            plt.xticks(rotation=45, fontsize=8)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

# 调用绘图函数
plot_metrics(metrics_df)