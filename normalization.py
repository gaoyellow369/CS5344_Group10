# 添加了mape的计算

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from aeon.regression.interval_based import TimeSeriesForestRegressor
from aeon.regression.interval_based import RandomIntervalRegressor
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.regression.interval_based import RandomIntervalSpectralEnsembleRegressor

# 数据预处理函数，调整 input_attributes.csv 中数据的时间顺序，添加剔除最后一列含非空数据的样本功能并显示剩余天数
def preprocess_data(input_file='input_attributes.csv', labels_file='labels.csv'):
    input_data = pd.read_csv(input_file, encoding='GBK')
    labels_data = pd.read_csv(labels_file, encoding='GBK', index_col=0)

    # 倒序排列 input_attributes.csv 中的数据，以使其与 labels.csv 对应
    input_data = input_data.iloc[::-1].reset_index(drop=True)

    # 获取最后一列 'anomaly' 数据用于判断
    last_column = input_data.iloc[:, -1]  # 获取最后一列
    input_data = input_data.iloc[:, :-1]  # 移除最后一列以外的数据

    # 确保 input_data 中的每一天都有完整的 24 行数据
    days = input_data.shape[0] // 24
    input_matrices = []
    labels_matrices = []

    # 转置 labels_data，使其每行对应一天的6个指标
    labels_data = labels_data.T
    labels_data.index = range(1, labels_data.shape[0] + 1)

    removed_days = []  # 用于存储剔除的天数
    for day in range(days):
        daily_data = input_data.iloc[day * 24:(day + 1) * 24, 1:25].values
        last_column_day = last_column.iloc[day * 24:(day + 1) * 24]  # 取出一天的最后一列数据

        # 检测该天数据是否包含 NaN，labels_data 含有空值，或者最后一列中任何一个值非空
        if np.isnan(daily_data).any() or labels_data.iloc[day].isnull().any() or last_column_day.notna().any():
            removed_days.append(day + 1)  # 存储剔除的天数（从1开始）
            continue
        input_matrices.append(daily_data)
        labels_matrices.append(labels_data.iloc[day].values)

    # 显示剔除的天数
    if removed_days:
        print(f"剔除的天数：{removed_days}")
    else:
        print("没有剔除任何天数的数据")

    # 显示剩余的天数
    remaining_days = len(input_matrices)
    print(f"剩余的天数：{remaining_days}")

    # 转换为 numpy 数组
    X = np.array(input_matrices)
    y = np.array(labels_matrices)
    return X, y

# 列归一化函数
def columnwise_normalization_fit(data, normalization_type):
    """
    对 24x24 的数据矩阵的每一列分别进行归一化，并返回拟合后的归一化器列表
    """    
    scalers = []
    scaler_class = MinMaxScaler if normalization_type == 'minmax' else StandardScaler
    if normalization_type is None:
        return data, None
    for i in range(data.shape[2]):
        scaler = scaler_class()
        scalers.append(scaler.fit(data[:, :, i]))
        data[:, :, i] = scalers[i].transform(data[:, :, i])
    return data, scalers

# 列归一化转换函数
def columnwise_normalization_transform(data, scalers):
    """
    使用在训练集上拟合的归一化器对 24x24 的数据矩阵每一列进行转换
    """
    if scalers is None:
        return data  # 如果没有归一化器，直接返回原始数据
    for i in range(data.shape[2]): # 遍历每一列（维度2是列数）
        data[:, :, i] = scalers[i].transform(data[:, :, i])
    return data

# 标签单独归一化函数
def normalize_labels_individually(y_train, normalization_type):
    """
    对 y_train 的每个输出单独进行归一化，并返回反归一化需要的 scaler 列表
    """
    scalers_y = []
    y_train_scaled = np.zeros_like(y_train)
    scaler_class = MinMaxScaler if normalization_type == 'minmax' else StandardScaler
    if normalization_type is None:
        return y_train, None   # 不进行归一化
    # 对每个输出（列）分别进行归一化
    for i in range(y_train.shape[1]):  # 遍历每个输出维度
        scaler = scaler_class() # 每个输出独立一个 scaler
        y_train_scaled[:, i] = scaler.fit_transform(y_train[:, i].reshape(-1, 1)).flatten()
        scalers_y.append(scaler)
    return y_train_scaled, scalers_y

# 反归一化函数
def inverse_transform_labels_individually(y_scaled, scalers_y):
    """
    对每个归一化后的 y 输出进行反归一化
    """
    if scalers_y is None:
        return y_scaled  # 如果没有归一化器，直接返回原始数据
    y_inversed = np.zeros_like(y_scaled)
    for i in range(y_scaled.shape[1]):
        y_inversed[:, i] = scalers_y[i].inverse_transform(y_scaled[:, i].reshape(-1, 1)).flatten()
    return y_inversed

# 计算评估指标函数
def calculate_metrics(y_true, y_pred):
    """
    计算 MSE, MAE, RMSE, R2 和 MAPE
    """
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    
    # 设置非常小的正数 epsilon 用于替换零值
    epsilon = 1e-10
    y_true_safe = np.where(y_true == 0, epsilon, y_true)  # 将 y_true 中的零值替换为 epsilon

    # 计算 MAPE
    mape = np.mean(np.abs((y_true_safe - y_pred) / y_true_safe), axis=0) * 100

    return mse, mae, rmse, r2, mape


# 训练模型函数，返回评估结果
def train_model(X_train, X_test, y_train, y_test, normalization_type=None, model=None, label_columns=None, exclude_columns=None, column_names=None):
    # 如果指定了要排除的列，删除这些列并打印列名
    if exclude_columns is not None and column_names is not None:
        excluded_column_names = [column_names[i] for i in exclude_columns]
        print(f"排除的列索引: {exclude_columns}")
        print(f"排除的列名: {excluded_column_names}")
        X_train = np.delete(X_train, exclude_columns, axis=2)
        X_test = np.delete(X_test, exclude_columns, axis=2)

    if label_columns is not None:
        y_train = y_train[:, label_columns]  # 选择指定列作为标签
        y_test = y_test[:, label_columns]  # 选择指定列作为测试标签

    # 进行归一化
    X_train_normalized, scalers = columnwise_normalization_fit(X_train, normalization_type)
    X_test_normalized = columnwise_normalization_transform(X_test, scalers)

    if isinstance(model, MultiOutputRegressor):
        # 检查 MultiOutputRegressor 中包装的模型是否为 RandomForestRegressor
        if isinstance(model.estimator, RandomForestRegressor):
            # 如果是包装的 RandomForestRegressor，则展平数据
            X_train_normalized = X_train_normalized.reshape(X_train_normalized.shape[0], -1)
            X_test_normalized = X_test_normalized.reshape(X_test_normalized.shape[0], -1)

    else:
        # 检查是否为单独的 RandomForestRegressor 模型
        if isinstance(model, RandomForestRegressor):
            # 如果是单独的 RandomForestRegressor，则展平数据
            X_train_normalized = X_train_normalized.reshape(X_train_normalized.shape[0], -1)
            X_test_normalized = X_test_normalized.reshape(X_test_normalized.shape[0], -1)

    # 判断是否使用 MultiOutputRegressor，如果是则一次性回归
    if isinstance(model, MultiOutputRegressor):

        # 对 y_train 进行归一化
        y_train_scaled, scalers_y = normalize_labels_individually(y_train, normalization_type)

        # 训练模型
        model.fit(X_train_normalized, y_train_scaled)

        # 预测训练集和测试集
        y_train_pred_scaled = model.predict(X_train_normalized)
        y_test_pred_scaled = model.predict(X_test_normalized)

        # 将预测结果反归一化
        y_train_pred = inverse_transform_labels_individually(y_train_pred_scaled, scalers_y)
        y_test_pred = inverse_transform_labels_individually(y_test_pred_scaled, scalers_y)

        # 计算训练集和测试集的评估指标
        train_mse, train_mae, train_rmse, train_r2, train_mape = calculate_metrics(y_train, y_train_pred)
        test_mse, test_mae, test_rmse, test_r2, test_mape = calculate_metrics(y_test, y_test_pred)
    
    # 如果不是 MultiOutputRegressor，则对每列分别进行回归
    else:
        # 初始化结果列表用于存储每列的结果
        y_train_preds = np.zeros_like(y_train)
        y_test_preds = np.zeros_like(y_test)

        # 对每一列单独训练模型并进行预测
        for i in range(y_train.shape[1]):
            # 对单列的 y 进行归一化
            y_train_col = y_train[:, i].reshape(-1, 1)
            y_train_col_scaled, scaler_y = normalize_labels_individually(y_train_col, normalization_type)

            # 创建模型并训练
            model.fit(X_train_normalized, y_train_col_scaled.ravel())

             # 预测训练集和测试集
            y_train_pred_scaled = model.predict(X_train_normalized).reshape(-1, 1)
            y_test_pred_scaled = model.predict(X_test_normalized).reshape(-1, 1)

            # 反归一化预测结果
            y_train_preds[:, i] = inverse_transform_labels_individually(y_train_pred_scaled, scaler_y).ravel()
            y_test_preds[:, i] = inverse_transform_labels_individually(y_test_pred_scaled, scaler_y).ravel()

        # 计算评估指标    
        train_mse, train_mae, train_rmse, train_r2, train_mape = calculate_metrics(y_train, y_train_preds)
        test_mse, test_mae, test_rmse, test_r2, test_mape = calculate_metrics(y_test, y_test_preds)

    return train_mse, train_mae, train_rmse, train_r2, train_mape, test_mse, test_mae, test_rmse, test_r2, test_mape

# 评估模型函数，针对每个标签单独记录指标
def evaluate_models(X_train, X_test, y_train, y_test, models, label_columns, exclude_columns=None, column_names=None):
    normalization_types = [None, 'minmax', 'standard']
    normalization_names = ['None', 'MinMax', 'Standard']

    metrics = {
        'Model': [], 
        'Normalization': [], 
        'Label': [],  # 用于存储不同标签的评估指标
        'Train_MSE': [], 'Test_MSE': [],
        'Train_MAE': [], 'Test_MAE': [],
        'Train_RMSE': [], 'Test_RMSE': [],
        'Train_R2': [], 'Test_R2': [],
        'Train_MAPE': [], 'Test_MAPE': []  # 新增 MAPE 列
    }

    for model_name, model in models.items():
        for norm_type, norm_name in zip(normalization_types, normalization_names):
            print(f"正在训练 {model_name} 模型，归一化类型: {norm_name}")
            train_mse, train_mae, train_rmse, train_r2, train_mape, test_mse, test_mae, test_rmse, test_r2, test_mape = train_model(
                X_train, X_test, y_train, y_test, norm_type, model, label_columns, exclude_columns, column_names
            )

            # 遍历每个标签的评估结果并分别存储
            for label_idx in range(train_mse.shape[0]):
                metrics['Model'].append(model_name)
                metrics['Normalization'].append(norm_name)
                metrics['Label'].append(f'Label {label_idx + 1}')
                metrics['Train_MSE'].append(train_mse[label_idx])
                metrics['Test_MSE'].append(test_mse[label_idx])
                metrics['Train_MAE'].append(train_mae[label_idx])
                metrics['Test_MAE'].append(test_mae[label_idx])
                metrics['Train_RMSE'].append(train_rmse[label_idx])
                metrics['Test_RMSE'].append(test_rmse[label_idx])
                metrics['Train_R2'].append(train_r2[label_idx])
                metrics['Test_R2'].append(test_r2[label_idx])
                metrics['Train_MAPE'].append(train_mape[label_idx])  # 新增 MAPE
                metrics['Test_MAPE'].append(test_mape[label_idx])  # 新增 MAPE

    return pd.DataFrame(metrics)

# 将模型名称按长度分为三部分的函数
def split_name_in_three_parts(name):
    length = len(name)
    part_size = length // 3
    if part_size == 0:
        return name  # 如果名称长度不足三部分，直接返回原名称
    
    part1 = name[:part_size]
    part2 = name[part_size:2 * part_size]
    part3 = name[2 * part_size:]
    
    return f"{part1}\n{part2}\n{part3}"

# 绘制评估指标图表的函数，显示每个标签的六条曲线
def plot_metrics(metrics_df):
    metrics = ['MSE', 'MAE', 'RMSE', 'R2', 'MAPE']  # 新增 MAPE
    normalization_types = ['None', 'MinMax', 'Standard']
    data_types = ['Train', 'Test']

    metrics_df['Model'] = metrics_df['Model'].apply(split_name_in_three_parts)

    for metric in metrics:
        for label in metrics_df['Label'].unique():  # 遍历每个标签
            plt.figure(figsize=(10, 6))
            for norm in normalization_types:
                for data_type in data_types:
                    label_name = f'{norm} {data_type} {metric}'
                    filtered_data = metrics_df[(metrics_df['Normalization'] == norm) & (metrics_df['Label'] == label)]
                    plt.plot(filtered_data['Model'], filtered_data[f'{data_type}_{metric}'], marker='o', label=label_name)

            plt.title(f'{metric} of {label} with Various Normalization')
            plt.xlabel('Model')
            plt.ylabel(metric)
            plt.legend()
            plt.xticks(rotation=0, fontsize=8)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

# 创建模型字典
models = {
    'RandomForestRegressor': RandomForestRegressor(n_estimators=10, random_state=42),
    'MultiOutput_RandomForestRegressor': MultiOutputRegressor(RandomForestRegressor(n_estimators=10, random_state=42)),
    'RandomIntervalRegressor': RandomIntervalRegressor(estimator=RandomForestRegressor(n_estimators=5), n_intervals=5, random_state=0),
    'MultiOutput_RandomIntervalRegressor': MultiOutputRegressor(RandomIntervalRegressor(estimator=RandomForestRegressor(n_estimators=5), n_intervals=5, random_state=0)),
    'RandomIntervalSpectralEnsembleRegressor': RandomIntervalSpectralEnsembleRegressor(n_estimators=10, random_state=0),
    'MultiOutput_RandomIntervalSpectralEnsembleRegressor': MultiOutputRegressor(RandomIntervalSpectralEnsembleRegressor(n_estimators=10, random_state=0)),
    'TimeSeriesForestRegressor': TimeSeriesForestRegressor(n_estimators=10, random_state=42),
    'MultiOutput_TimeSeriesForestRegressor': MultiOutputRegressor(TimeSeriesForestRegressor(n_estimators=10, random_state=42)),
    'KNeighborsTimeSeriesRegressor': KNeighborsTimeSeriesRegressor(distance="euclidean"),
    'MultiOutput_KNeighborsTimeSeriesRegressor': MultiOutputRegressor(KNeighborsTimeSeriesRegressor(distance="euclidean")) 
}

# 调用预处理函数
input_file = 'input_attributes.csv'
labels_file = 'labels.csv'
X, y = preprocess_data(input_file, labels_file)

# 获取 input_attributes.csv 的列名（假设前24列是特征）
column_names = pd.read_csv(input_file, encoding='GBK').columns[1:25]  # 假设第0列不是特征

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 指定标签列
label_columns = [1, 2, 3, 4, 5]

# 评估模型 exclude_columns可以指定x中指定索引列不参与训练和测试，范围为0到23
metrics_df = evaluate_models(X_train, X_test, y_train, y_test, models, label_columns, exclude_columns=[10,12,21,23], column_names=column_names)

# 绘制评估图表
plot_metrics(metrics_df)
