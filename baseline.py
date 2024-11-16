# 只保留时间序列随机森林，归一化选择标准归一化

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from aeon.regression.interval_based import TimeSeriesForestRegressor

# 数据预处理函数
def preprocess_data(input_file='input_attributes.csv', labels_file='labels.csv'):
    input_data = pd.read_csv(input_file, encoding='GBK')
    labels_data = pd.read_csv(labels_file, encoding='GBK', index_col=0)

    # 倒序排列 input_attributes.csv 中的数据，以使其与 labels.csv 对应
    input_data = input_data.iloc[::-1].reset_index(drop=True)

    # 获取最后一列 'anomaly' 数据用于判断
    last_column = input_data.iloc[:, -1]
    input_data = input_data.iloc[:, :-1]

    # 确保 input_data 中的每一天都有完整的 24 行数据
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
    return X, y

# 列归一化函数
def columnwise_normalization_fit(data):
    scalers = []
    for i in range(data.shape[2]):
        scaler = StandardScaler()
        scalers.append(scaler.fit(data[:, :, i]))
        data[:, :, i] = scalers[i].transform(data[:, :, i])
    return data, scalers

# 列归一化转换函数
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

# 训练模型函数
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

    X_train_normalized, scalers = columnwise_normalization_fit(X_train)
    X_test_normalized = columnwise_normalization_transform(X_test, scalers)

    train_results = []
    test_results = []

    for i in range(y_train.shape[1]):
        model = TimeSeriesForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train_normalized, y_train[:, i])

        y_train_pred = model.predict(X_train_normalized)
        y_test_pred = model.predict(X_test_normalized)

        train_metrics = calculate_metrics(y_train[:, i], y_train_pred)
        test_metrics = calculate_metrics(y_test[:, i], y_test_pred)

        train_results.append(train_metrics)
        test_results.append(test_metrics)

    return train_results, test_results

# 评估模型函数
def evaluate_model(X_train, X_test, y_train, y_test, label_columns, exclude_columns=None, column_names=None):
    train_metrics, test_metrics = train_model(X_train, X_test, y_train, y_test, label_columns, exclude_columns, column_names)

    metrics = {
        'Label': [],
        'Train_MSE': [], 'Test_MSE': [],
        'Train_MAE': [], 'Test_MAE': [],
        'Train_RMSE': [], 'Test_RMSE': [],
        'Train_R2': [], 'Test_R2': [],
        'Train_MAPE': [], 'Test_MAPE': []
    }

    for i, (train, test) in enumerate(zip(train_metrics, test_metrics)):
        metrics['Label'].append(f'Label {i + 1}')
        metrics['Train_MSE'].append(train[0])
        metrics['Test_MSE'].append(test[0])
        metrics['Train_MAE'].append(train[1])
        metrics['Test_MAE'].append(test[1])
        metrics['Train_RMSE'].append(train[2])
        metrics['Test_RMSE'].append(test[2])
        metrics['Train_R2'].append(train[3])
        metrics['Test_R2'].append(test[3])
        metrics['Train_MAPE'].append(train[4])
        metrics['Test_MAPE'].append(test[4])

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
label_columns = [1, 2, 3, 4, 5]

# 评估模型
metrics_df = evaluate_model(X_train, X_test, y_train, y_test, label_columns, exclude_columns=[10, 12, 21, 23], column_names=column_names)

# 输出结果
print(metrics_df)
