import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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

    # 转置 labels_data，使其每行对应一天的6个指标
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

# 模拟退火算法用于特征选择
class SimulatedAnnealingFeatureSelector:
    def __init__(self, model, X, y, initial_temperature=100, cooling_rate=0.95, stopping_temperature=1, max_iterations=100):
        self.model = model
        self.X = X
        self.y = y
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature
        self.max_iterations = max_iterations
        self.best_features = np.arange(X.shape[1])
        self.best_score = float('inf')
        self.scores_history = []

    def evaluate_model(self, selected_features):
        X_subset = self.X[:, selected_features]
        self.model.fit(X_subset, self.y)
        y_pred = self.model.predict(X_subset)
        mse = mean_squared_error(self.y, y_pred)
        return mse

    def run(self):
        current_temperature = self.initial_temperature
        current_features = copy.deepcopy(self.best_features)
        current_score = self.evaluate_model(current_features)
        self.scores_history.append(current_score)

        while current_temperature > self.stopping_temperature:
            for _ in range(self.max_iterations):
                new_features = self._generate_new_features(current_features)
                new_score = self.evaluate_model(new_features)

                if new_score < current_score or np.exp((current_score - new_score) / current_temperature) > np.random.rand():
                    current_features = new_features
                    current_score = new_score

                    if new_score < self.best_score:
                        self.best_features = new_features
                        self.best_score = new_score

                self.scores_history.append(new_score)

            current_temperature *= self.cooling_rate
            print(f"Current temperature: {current_temperature:.2f}, Best MSE: {self.best_score:.4f}")

        self.plot_score_history()
        print(f"Best feature subset: {self.best_features}")
        print(f"Best MSE: {self.best_score:.4f}")
        return self.best_features

    def _generate_new_features(self, current_features):
        new_features = current_features.copy()
        if len(new_features) > 1 and np.random.rand() > 0.5:
            remove_idx = np.random.randint(0, len(new_features))
            new_features = np.delete(new_features, remove_idx)
        else:
            remaining_features = np.setdiff1d(np.arange(self.X.shape[1]), new_features)
            if len(remaining_features) > 0:
                add_feature = np.random.choice(remaining_features)
                new_features = np.append(new_features, add_feature)
        return new_features

    def plot_score_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.scores_history, marker='o', linestyle='-', color='b')
        plt.title('Simulated Annealing Score History')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.show()

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

# 针对每个目标标签分别进行模拟退火特征提取，并排除指定列
def run_feature_selection_per_label(X_train, y_train, model, label_columns, exclude_columns=None):
    if exclude_columns is not None:
        print(f"排除的列索引: {exclude_columns}")
        X_train = np.delete(X_train, exclude_columns, axis=2)

    selected_features_per_label = []
    for label_idx in label_columns:
        print(f"\nRunning simulated annealing for Label {label_idx + 1}")
        y_train_single = y_train[:, label_idx]  # 提取单个标签
        selector = SimulatedAnnealingFeatureSelector(model=model, X=X_train.reshape(X_train.shape[0], -1), y=y_train_single)
        selected_features = selector.run()
        selected_features_per_label.append(selected_features)
        print(f"Selected features for Label {label_idx + 1}: {selected_features}")
    return selected_features_per_label

# 调用预处理函数
input_file = 'input_attributes.csv'
labels_file = 'labels.csv'
X, y = preprocess_data(input_file, labels_file)

# 获取列名
column_names = pd.read_csv(input_file, encoding='GBK').columns[1:25]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 指定标签列和排除的特征列
label_columns = [1, 2, 3, 4,5]  # 指定目标标签索引
exclude_columns = [10, 12, 21, 23]  # 指定要排除的输入特征列

# 初始化模型
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)

# 对每个目标标签分别运行模拟退火特征提取
selected_features_per_label = run_feature_selection_per_label(X_train, y_train, rf_model, label_columns, exclude_columns)
