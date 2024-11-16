#自定义链式回归

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

# 自定义回归链训练函数
def train_regressor_chain(X_train, X_test, y_train, y_test, label_columns=None, exclude_columns=None, column_names=None):
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

    num_labels = y_train.shape[1]
    y_train_predictions = np.zeros_like(y_train)
    y_test_predictions = np.zeros_like(y_test)

    for round_num in range(2):  # 进行两轮训练
        print(f"Training round {round_num + 1}")
        for i in range(num_labels):
            model = TimeSeriesForestRegressor(n_estimators=10, random_state=42)

            # 构建输入特征矩阵
            if round_num == 0:
                if i == 0:
                    X_train_augmented = X_train_normalized
                    X_test_augmented = X_test_normalized
                else:
                    y_train_pred_expanded = np.repeat(y_train_predictions[:, :i][:, np.newaxis, :], X_train.shape[1], axis=1)
                    y_test_pred_expanded = np.repeat(y_test_predictions[:, :i][:, np.newaxis, :], X_test.shape[1], axis=1)
                    X_train_augmented = np.concatenate([X_train_normalized, y_train_pred_expanded], axis=2)
                    X_test_augmented = np.concatenate([X_test_normalized, y_test_pred_expanded], axis=2)
            else:
                y_train_pred_expanded = np.repeat(y_train_predictions[:, (i + 1):][:, np.newaxis, :], X_train.shape[1], axis=1)
                y_test_pred_expanded = np.repeat(y_test_predictions[:, (i + 1):][:, np.newaxis, :], X_test.shape[1], axis=1)
                X_train_augmented = np.concatenate([X_train_normalized, y_train_pred_expanded], axis=2)
                X_test_augmented = np.concatenate([X_test_normalized, y_test_pred_expanded], axis=2)

            model.fit(X_train_augmented, y_train[:, i])

            y_train_pred = model.predict(X_train_augmented)
            y_test_pred = model.predict(X_test_augmented)

            y_train_predictions[:, i] = y_train_pred
            y_test_predictions[:, i] = y_test_pred

            train_metrics = calculate_metrics(y_train[:, i], y_train_pred)
            test_metrics = calculate_metrics(y_test[:, i], y_test_pred)

            # 使用科学计数法打印结果
            print(f"Label {i + 1}: Train MSE: {train_metrics[0]:.3e}, Test MSE: {test_metrics[0]:.3e}")
            print(f"Label {i + 1}: Train MAE: {train_metrics[1]:.3e}, Test MAE: {test_metrics[1]:.3e}")
            print(f"Label {i + 1}: Train RMSE: {train_metrics[2]:.3e}, Test RMSE: {test_metrics[2]:.3e}")
            print(f"Label {i + 1}: Train R2: {train_metrics[3]:.3e}, Test R2: {test_metrics[3]:.3e}")
            print(f"Label {i + 1}: Train MAPE: {train_metrics[4]:.3e}, Test MAPE: {test_metrics[4]:.3e}")

    return y_train_predictions, y_test_predictions

# 调用预处理函数
input_file = 'input_attributes.csv'
labels_file = 'labels.csv'
X, y = preprocess_data(input_file, labels_file)

# 获取列名
column_names = pd.read_csv(input_file, encoding='GBK').columns[1:25]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 指定标签列
label_columns = [1, 2, 3, 4, 5]  # 修正标签列索引从 0 开始

# 训练并评估模型
train_regressor_chain(X_train, X_test, y_train, y_test, label_columns, exclude_columns=[10, 12, 21, 23], column_names=column_names)