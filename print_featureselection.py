import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from aeon.regression.interval_based import TimeSeriesForestRegressor
import re
import os

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

# 从打印结果文本中解析 temperature 和 MSE
def parse_results_from_text(print_text):
    temperatures = []
    mses = []
    lines = print_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        match = re.match(r"Current temperature: ([0-9\.]+), Best MSE: ([0-9\.]+)", line)
        if match:
            temperatures.append(float(match.group(1)))
            mses.append(float(match.group(2)))
        else:
            print(f"未能解析的行: {line}")
    return temperatures, mses

# 修改基准模型函数以使用指定的基准线数值
def baseline_model(baseline_values, label_idx):
    baseline_value = baseline_values[label_idx]
    print(f"Using specified baseline MSE for Label {label_idx + 1}: {baseline_value:.4f}")
    return baseline_value

# 绘制图像并添加多条基准线
def plot_results_with_baseline(temperatures, mses, baseline_mses, label_idx):
    if len(temperatures) > 1 and len(mses) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(temperatures, mses, marker='o', linestyle='-', color='b', label='Feature Selection (Simulated annealing)')

        # 绘制每条基准线，使用指定的颜色
        for baseline in baseline_mses:
            value = baseline['value']
            name = baseline['name']
            color = baseline['color']  # 使用指定的颜色

            plt.axhline(y=value, linestyle='--', color=color, label=f'{name} (MSE: {value:.4f})')

        plt.title(f'MSE for Label {label_idx}')
        plt.xlabel('Temperature')
        plt.ylabel('MSE')
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.legend()
        output_dir = 'output_images'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/temperature_vs_mse_label_{label_idx + 1}.png')
        plt.close()
        print(f"图像已保存为 {output_dir}/temperature_vs_mse_label_{label_idx + 1}.png")
    else:
        print(f"Label {label_idx + 1}: 温度和 MSE 数据不足，无法绘制完整曲线。")


# 调用预处理函数
input_file = 'input_attributes.csv'
labels_file = 'labels.csv'
X, y = preprocess_data(input_file, labels_file)

# 获取列名
column_names = pd.read_csv(input_file, encoding='GBK').columns[1:25]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 指定标签列和排除的特征列
label_columns = [1, 2, 3, 4, 5]  # 指定目标标签索引
exclude_columns = [10, 12, 21, 23]  # 指定要排除的输入特征列

# 指定的基准线数值，每个标签有不同的值
#specified_baseline_values = [0, 1.584, 0.607, 204.449, 286.672, 3.697]  # 可以根据需要修改这些值

# 每个标签的多条基准线
baseline_mses_list = {
    1: [{'value': 1.584, 'name': 'Baseline (no_featureselection)', 'color': 'green'}, 
        {'value': 1.2848, 'name': 'SelectKBest', 'color': 'red'}],
    2: [{'value': 0.607, 'name': 'Baseline (no_featureselection)', 'color': 'green'}, 
        {'value': 0.5999, 'name': 'SelectKBest', 'color': 'red'}],
    3: [{'value': 204.449, 'name': 'Baseline (no_featureselection)', 'color': 'green'}, 
        {'value': 218.431, 'name': 'SelectKBest', 'color': 'red'}],
    4: [{'value': 286.672, 'name': 'Baseline (no_featureselection)', 'color': 'green'}, 
        {'value': 205.2634, 'name': 'SelectKBest', 'color': 'red'}],
    5: [{'value': 3.697, 'name': 'Baseline (no_featureselection)', 'color': 'green'}, 
        {'value': 2.6845, 'name': 'SelectKBest', 'color': 'red'}],
}



# 打印结果文本示例
print_texts = [
    """Current temperature: 95.00, Best MSE: 1.3772
    Current temperature: 90.25, Best MSE: 1.3758
    Current temperature: 85.74, Best MSE: 1.3241
    Current temperature: 81.45, Best MSE: 1.3241
    Current temperature: 77.38, Best MSE: 1.2740
    Current temperature: 73.51, Best MSE: 1.2740
    Current temperature: 69.83, Best MSE: 1.2740
    Current temperature: 66.34, Best MSE: 1.2740
    Current temperature: 63.02, Best MSE: 1.2740
    Current temperature: 59.87, Best MSE: 1.2740
    Current temperature: 56.88, Best MSE: 1.2740
    Current temperature: 54.04, Best MSE: 1.2740
    Current temperature: 51.33, Best MSE: 1.2740
    Current temperature: 48.77, Best MSE: 1.2740
    Current temperature: 46.33, Best MSE: 1.2740
    Current temperature: 44.01, Best MSE: 1.2740
    Current temperature: 41.81, Best MSE: 1.2740
    Current temperature: 39.72, Best MSE: 1.2740
    Current temperature: 37.74, Best MSE: 1.2740
    Current temperature: 35.85, Best MSE: 1.2740
    Current temperature: 34.06, Best MSE: 1.2740
    Current temperature: 32.35, Best MSE: 1.2740
    Current temperature: 30.74, Best MSE: 1.2740
    Current temperature: 29.20, Best MSE: 1.2740
    Current temperature: 27.74, Best MSE: 1.2740
    Current temperature: 26.35, Best MSE: 1.2740
    Current temperature: 25.03, Best MSE: 1.2740
    Current temperature: 23.78, Best MSE: 1.2740
    Current temperature: 22.59, Best MSE: 1.2740
    Current temperature: 21.46, Best MSE: 1.2740
    Current temperature: 20.39, Best MSE: 1.2740
    Current temperature: 19.37, Best MSE: 1.2740
    Current temperature: 18.40, Best MSE: 1.2740
    Current temperature: 17.48, Best MSE: 1.2740
    Current temperature: 16.61, Best MSE: 1.2740
    Current temperature: 15.78, Best MSE: 1.2740
    Current temperature: 14.99, Best MSE: 1.2740
    Current temperature: 14.24, Best MSE: 1.2740
    Current temperature: 13.53, Best MSE: 1.2740
    Current temperature: 12.85, Best MSE: 1.2740
    Current temperature: 12.21, Best MSE: 1.2740
    Current temperature: 11.60, Best MSE: 1.2740
    Current temperature: 11.02, Best MSE: 1.2740
    Current temperature: 10.47, Best MSE: 1.2740
    Current temperature: 9.94, Best MSE: 1.2740
    Current temperature: 9.45, Best MSE: 1.2740
    Current temperature: 8.97, Best MSE: 1.2740
    Current temperature: 8.53, Best MSE: 1.2740
    Current temperature: 8.10, Best MSE: 1.2740
    Current temperature: 7.69, Best MSE: 1.2740
    Current temperature: 7.31, Best MSE: 1.2740
    Current temperature: 6.94, Best MSE: 1.2740
    Current temperature: 6.60, Best MSE: 1.2740
    Current temperature: 6.27, Best MSE: 1.2740
    Current temperature: 5.95, Best MSE: 1.2740
    Current temperature: 5.66, Best MSE: 1.2740
    Current temperature: 5.37, Best MSE: 1.2740
    Current temperature: 5.10, Best MSE: 1.2740
    Current temperature: 4.85, Best MSE: 1.2740
    Current temperature: 4.61, Best MSE: 1.2740
    Current temperature: 4.38, Best MSE: 1.2740
    Current temperature: 4.16, Best MSE: 1.2740
    Current temperature: 3.95, Best MSE: 1.2740
    Current temperature: 3.75, Best MSE: 1.2740
    Current temperature: 3.56, Best MSE: 1.2740
    Current temperature: 3.39, Best MSE: 1.2740
    Current temperature: 3.22, Best MSE: 1.2740
    Current temperature: 3.06, Best MSE: 1.2740
    Current temperature: 2.90, Best MSE: 1.2740
    Current temperature: 2.76, Best MSE: 1.2740
    Current temperature: 2.62, Best MSE: 1.2740
    Current temperature: 5.37, Best MSE: 1.2740
    Current temperature: 5.10, Best MSE: 1.2740
    Current temperature: 4.85, Best MSE: 1.2740
    Current temperature: 4.61, Best MSE: 1.2740
    Current temperature: 4.38, Best MSE: 1.2740
    Current temperature: 4.16, Best MSE: 1.2740
    Current temperature: 3.95, Best MSE: 1.2740
    Current temperature: 3.75, Best MSE: 1.2740
    Current temperature: 3.56, Best MSE: 1.2740
    Current temperature: 3.39, Best MSE: 1.2740
    Current temperature: 3.22, Best MSE: 1.2740
    Current temperature: 3.06, Best MSE: 1.2740
    Current temperature: 2.90, Best MSE: 1.2740
    Current temperature: 2.76, Best MSE: 1.2740
    Current temperature: 2.62, Best MSE: 1.2740
    Current temperature: 4.38, Best MSE: 1.2740
    Current temperature: 4.16, Best MSE: 1.2740
    Current temperature: 3.95, Best MSE: 1.2740
    Current temperature: 3.75, Best MSE: 1.2740
    Current temperature: 3.56, Best MSE: 1.2740
    Current temperature: 3.39, Best MSE: 1.2740
    Current temperature: 3.22, Best MSE: 1.2740
    Current temperature: 3.06, Best MSE: 1.2740
    Current temperature: 2.90, Best MSE: 1.2740
    Current temperature: 2.76, Best MSE: 1.2740
    Current temperature: 2.62, Best MSE: 1.2740
    Current temperature: 3.56, Best MSE: 1.2740
    Current temperature: 3.39, Best MSE: 1.2740
    Current temperature: 3.22, Best MSE: 1.2740
    Current temperature: 3.06, Best MSE: 1.2740
    Current temperature: 2.90, Best MSE: 1.2740
    Current temperature: 2.76, Best MSE: 1.2740
    Current temperature: 2.62, Best MSE: 1.2740
    Current temperature: 2.90, Best MSE: 1.2740
    Current temperature: 2.76, Best MSE: 1.2740
    Current temperature: 2.62, Best MSE: 1.2740
    Current temperature: 2.62, Best MSE: 1.2740
    Current temperature: 2.49, Best MSE: 1.2740
    Current temperature: 2.36, Best MSE: 1.2740
    Current temperature: 2.25, Best MSE: 1.2740
    Current temperature: 2.13, Best MSE: 1.2740
    Current temperature: 2.03, Best MSE: 1.2740
    Current temperature: 1.93, Best MSE: 1.2740
    Current temperature: 1.83, Best MSE: 1.2740
    Current temperature: 1.74, Best MSE: 1.2740
    Current temperature: 1.65, Best MSE: 1.2740
    Current temperature: 1.57, Best MSE: 1.2740
    Current temperature: 1.49, Best MSE: 1.2740
    Current temperature: 1.42, Best MSE: 1.2740
    Current temperature: 1.35, Best MSE: 1.2740
    Current temperature: 1.28, Best MSE: 1.2740
    Current temperature: 1.21, Best MSE: 1.2740
    Current temperature: 1.15, Best MSE: 1.2740
    Current temperature: 1.10, Best MSE: 1.2740
    Current temperature: 1.04, Best MSE: 1.2740
    Current temperature: 0.99, Best MSE: 1.2740""",
    """Current temperature: 95.00, Best MSE: 0.5439
    Current temperature: 90.25, Best MSE: 0.5213
    Current temperature: 85.74, Best MSE: 0.5211
    Current temperature: 81.45, Best MSE: 0.5211
    Current temperature: 77.38, Best MSE: 0.5211
    Current temperature: 73.51, Best MSE: 0.5211
    Current temperature: 69.83, Best MSE: 0.5211
    Current temperature: 66.34, Best MSE: 0.5211
    Current temperature: 63.02, Best MSE: 0.5211
    Current temperature: 59.87, Best MSE: 0.5211
    Current temperature: 56.88, Best MSE: 0.4951
    Current temperature: 54.04, Best MSE: 0.4951
    Current temperature: 51.33, Best MSE: 0.4951
    Current temperature: 48.77, Best MSE: 0.4951
    Current temperature: 46.33, Best MSE: 0.4951
    Current temperature: 44.01, Best MSE: 0.4951
    Current temperature: 41.81, Best MSE: 0.4951
    Current temperature: 39.72, Best MSE: 0.4951
    Current temperature: 37.74, Best MSE: 0.4951
    Current temperature: 35.85, Best MSE: 0.4951
    Current temperature: 34.06, Best MSE: 0.4951
    Current temperature: 32.35, Best MSE: 0.4951
    Current temperature: 30.74, Best MSE: 0.4951
    Current temperature: 29.20, Best MSE: 0.4951
    Current temperature: 27.74, Best MSE: 0.4951
    Current temperature: 26.35, Best MSE: 0.4951
    Current temperature: 25.03, Best MSE: 0.4951
    Current temperature: 23.78, Best MSE: 0.4951
    Current temperature: 22.59, Best MSE: 0.4951
    Current temperature: 21.46, Best MSE: 0.4951
    Current temperature: 20.39, Best MSE: 0.4951
    Current temperature: 19.37, Best MSE: 0.4951
    Current temperature: 18.40, Best MSE: 0.4951
    Current temperature: 17.48, Best MSE: 0.4951
    Current temperature: 16.61, Best MSE: 0.4951
    Current temperature: 15.78, Best MSE: 0.4951
    Current temperature: 14.99, Best MSE: 0.4951
    Current temperature: 14.24, Best MSE: 0.4951
    Current temperature: 13.53, Best MSE: 0.4951
    Current temperature: 12.85, Best MSE: 0.4951
    Current temperature: 12.21, Best MSE: 0.4951
    Current temperature: 11.60, Best MSE: 0.4951
    Current temperature: 11.02, Best MSE: 0.4951
    Current temperature: 10.47, Best MSE: 0.4951
    Current temperature: 9.94, Best MSE: 0.4951
    Current temperature: 9.45, Best MSE: 0.4951
    Current temperature: 8.97, Best MSE: 0.4951
    Current temperature: 8.53, Best MSE: 0.4951
    Current temperature: 8.10, Best MSE: 0.4951
    Current temperature: 7.69, Best MSE: 0.4951
    Current temperature: 7.31, Best MSE: 0.4951
    Current temperature: 6.94, Best MSE: 0.4951
    Current temperature: 6.60, Best MSE: 0.4951
    Current temperature: 6.27, Best MSE: 0.4951
    Current temperature: 5.95, Best MSE: 0.4951
    Current temperature: 5.66, Best MSE: 0.4951
    Current temperature: 5.37, Best MSE: 0.4951
    Current temperature: 5.10, Best MSE: 0.4951
    Current temperature: 4.85, Best MSE: 0.4951
    Current temperature: 4.61, Best MSE: 0.4951
    Current temperature: 4.38, Best MSE: 0.4951
    Current temperature: 4.16, Best MSE: 0.4951
    Current temperature: 3.95, Best MSE: 0.4951
    Current temperature: 3.75, Best MSE: 0.4951
    Current temperature: 3.56, Best MSE: 0.4951
    Current temperature: 3.39, Best MSE: 0.4951
    Current temperature: 3.22, Best MSE: 0.4951
    Current temperature: 3.06, Best MSE: 0.4951
    Current temperature: 2.90, Best MSE: 0.4951
    Current temperature: 2.76, Best MSE: 0.4951
    Current temperature: 2.62, Best MSE: 0.4951
    Current temperature: 2.49, Best MSE: 0.4951
    Current temperature: 2.36, Best MSE: 0.4951
    Current temperature: 2.25, Best MSE: 0.4951
    Current temperature: 2.13, Best MSE: 0.4951
    Current temperature: 2.03, Best MSE: 0.4951
    Current temperature: 1.93, Best MSE: 0.4951
    Current temperature: 1.83, Best MSE: 0.4951
    Current temperature: 1.74, Best MSE: 0.4951
    Current temperature: 1.65, Best MSE: 0.4951
    Current temperature: 1.57, Best MSE: 0.4951
    Current temperature: 1.49, Best MSE: 0.4951
    Current temperature: 1.42, Best MSE: 0.4951
    Current temperature: 1.35, Best MSE: 0.4951
    Current temperature: 1.28, Best MSE: 0.4951
    Current temperature: 1.21, Best MSE: 0.4951
    Current temperature: 1.15, Best MSE: 0.4951
    Current temperature: 1.10, Best MSE: 0.4951
    Current temperature: 1.04, Best MSE: 0.4951
    Current temperature: 0.99, Best MSE: 0.4951""",
    """Current temperature: 95.00, Best MSE: 194.5409
    Current temperature: 90.25, Best MSE: 194.5409
    Current temperature: 85.74, Best MSE: 192.7564
    Current temperature: 81.45, Best MSE: 182.1000
    Current temperature: 77.38, Best MSE: 182.1000
    Current temperature: 73.51, Best MSE: 182.1000
    Current temperature: 69.83, Best MSE: 182.1000
    Current temperature: 66.34, Best MSE: 182.1000
    Current temperature: 63.02, Best MSE: 182.1000
    Current temperature: 59.87, Best MSE: 182.1000
    Current temperature: 56.88, Best MSE: 182.1000
    Current temperature: 54.04, Best MSE: 182.1000
    Current temperature: 51.33, Best MSE: 182.1000
    Current temperature: 48.77, Best MSE: 182.1000
    Current temperature: 46.33, Best MSE: 182.1000
    Current temperature: 44.01, Best MSE: 182.1000
    Current temperature: 41.81, Best MSE: 182.1000
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 32.35, Best MSE: 181.2747
    Current temperature: 30.74, Best MSE: 181.2747
    Current temperature: 29.20, Best MSE: 181.2747
    Current temperature: 27.74, Best MSE: 181.2747
    Current temperature: 26.35, Best MSE: 181.2747
    Current temperature: 25.03, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 32.35, Best MSE: 181.2747
    Current temperature: 30.74, Best MSE: 181.2747
    Current temperature: 29.20, Best MSE: 181.2747
    Current temperature: 27.74, Best MSE: 181.2747
    Current temperature: 26.35, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 32.35, Best MSE: 181.2747
    Current temperature: 30.74, Best MSE: 181.2747
    Current temperature: 29.20, Best MSE: 181.2747
    Current temperature: 27.74, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 32.35, Best MSE: 181.2747
    Current temperature: 30.74, Best MSE: 181.2747
    Current temperature: 29.20, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 32.35, Best MSE: 181.2747
    Current temperature: 30.74, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 32.35, Best MSE: 181.2747
    Current temperature: 30.74, Best MSE: 181.2747
    Current temperature: 29.20, Best MSE: 181.2747
    Current temperature: 27.74, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 32.35, Best MSE: 181.2747
    Current temperature: 30.74, Best MSE: 181.2747
    Current temperature: 29.20, Best MSE: 181.2747
    Current temperature: 27.74, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 32.35, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 39.72, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 37.74, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 35.85, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 34.06, Best MSE: 181.2747
    Current temperature: 32.35, Best MSE: 181.2747
    Current temperature: 30.74, Best MSE: 181.2747
    Current temperature: 29.20, Best MSE: 181.2747
    Current temperature: 27.74, Best MSE: 181.2747
    Current temperature: 26.35, Best MSE: 181.2747
    Current temperature: 25.03, Best MSE: 181.2747
    Current temperature: 23.78, Best MSE: 181.2747
    Current temperature: 22.59, Best MSE: 181.2747
    Current temperature: 21.46, Best MSE: 181.2747
    Current temperature: 20.39, Best MSE: 181.2747
    Current temperature: 19.37, Best MSE: 181.2747
    Current temperature: 18.40, Best MSE: 181.2747
    Current temperature: 22.59, Best MSE: 181.2747
    Current temperature: 21.46, Best MSE: 181.2747
    Current temperature: 20.39, Best MSE: 181.2747
    Current temperature: 19.37, Best MSE: 181.2747
    Current temperature: 18.40, Best MSE: 181.2747
    Current temperature: 21.46, Best MSE: 181.2747
    Current temperature: 20.39, Best MSE: 181.2747
    Current temperature: 19.37, Best MSE: 181.2747
    Current temperature: 18.40, Best MSE: 181.2747
    Current temperature: 20.39, Best MSE: 181.2747
    Current temperature: 19.37, Best MSE: 181.2747
    Current temperature: 18.40, Best MSE: 181.2747
    Current temperature: 17.48, Best MSE: 181.2747
    Current temperature: 19.37, Best MSE: 181.2747
    Current temperature: 18.40, Best MSE: 181.2747
    Current temperature: 17.48, Best MSE: 181.2747
    Current temperature: 16.61, Best MSE: 181.2747
    Current temperature: 18.40, Best MSE: 181.2747
    Current temperature: 17.48, Best MSE: 181.2747
    Current temperature: 16.61, Best MSE: 181.2747
    Current temperature: 15.78, Best MSE: 181.2747
    Current temperature: 17.48, Best MSE: 181.2747
    Current temperature: 16.61, Best MSE: 181.2747
    Current temperature: 15.78, Best MSE: 181.2747
    Current temperature: 16.61, Best MSE: 181.2747
    Current temperature: 15.78, Best MSE: 181.2747
    Current temperature: 15.78, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 13.53, Best MSE: 181.2747
    Current temperature: 12.85, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 13.53, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 13.53, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 13.53, Best MSE: 181.2747
    Current temperature: 12.85, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 13.53, Best MSE: 181.2747
    Current temperature: 12.85, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 13.53, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 13.53, Best MSE: 181.2747
    Current temperature: 12.85, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 13.53, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.99, Best MSE: 181.2747
    Current temperature: 14.24, Best MSE: 181.2747
    Current temperature: 13.53, Best MSE: 181.2747
    Current temperature: 12.85, Best MSE: 181.2747
    Current temperature: 12.21, Best MSE: 181.2747
    Current temperature: 11.60, Best MSE: 181.2747
    Current temperature: 11.02, Best MSE: 181.2747
    Current temperature: 10.47, Best MSE: 181.2747
    Current temperature: 9.94, Best MSE: 181.2747
    Current temperature: 9.45, Best MSE: 181.2747
    Current temperature: 8.97, Best MSE: 181.2747
    Current temperature: 8.53, Best MSE: 181.2747
    Current temperature: 8.10, Best MSE: 181.2747
    Current temperature: 7.69, Best MSE: 181.2747
    Current temperature: 7.31, Best MSE: 181.2747
    Current temperature: 6.94, Best MSE: 181.2747
    Current temperature: 6.60, Best MSE: 181.2747
    Current temperature: 6.27, Best MSE: 181.2747
    Current temperature: 5.95, Best MSE: 181.2747
    Current temperature: 5.66, Best MSE: 181.2747
    Current temperature: 5.37, Best MSE: 181.2747
    Current temperature: 5.10, Best MSE: 181.2747
    Current temperature: 4.85, Best MSE: 179.8646
    Current temperature: 4.61, Best MSE: 179.8646
    Current temperature: 4.38, Best MSE: 179.8646
    Current temperature: 4.16, Best MSE: 179.8646
    Current temperature: 3.95, Best MSE: 174.9400
    Current temperature: 3.75, Best MSE: 174.9400
    Current temperature: 3.56, Best MSE: 174.9400
    Current temperature: 3.39, Best MSE: 174.9400
    Current temperature: 3.22, Best MSE: 174.9400
    Current temperature: 3.06, Best MSE: 174.9400
    Current temperature: 2.90, Best MSE: 174.9400
    Current temperature: 2.76, Best MSE: 174.9400
    Current temperature: 2.62, Best MSE: 174.9400
    Current temperature: 2.49, Best MSE: 174.9400
    Current temperature: 2.36, Best MSE: 174.9400
    Current temperature: 2.25, Best MSE: 174.9400
    Current temperature: 2.13, Best MSE: 174.9400
    Current temperature: 2.03, Best MSE: 174.9400
    Current temperature: 1.93, Best MSE: 174.9400
    Current temperature: 1.83, Best MSE: 174.9400
    Current temperature: 1.74, Best MSE: 174.9400
    Current temperature: 1.65, Best MSE: 174.9400
    Current temperature: 1.57, Best MSE: 174.9400
    Current temperature: 1.49, Best MSE: 174.9400
    Current temperature: 1.42, Best MSE: 174.9400
    Current temperature: 1.35, Best MSE: 174.9400
    Current temperature: 1.28, Best MSE: 174.9400
    Current temperature: 1.21, Best MSE: 174.9400
    Current temperature: 1.15, Best MSE: 174.9400
    Current temperature: 1.10, Best MSE: 174.9400
    Current temperature: 1.04, Best MSE: 174.9400
    Current temperature: 0.99, Best MSE: 174.9400""",
    """Current temperature: 95.00, Best MSE: 253.9609
    Current temperature: 90.25, Best MSE: 246.0233
    Current temperature: 85.74, Best MSE: 246.0233
    Current temperature: 81.45, Best MSE: 236.1671
    Current temperature: 77.38, Best MSE: 236.1671
    Current temperature: 73.51, Best MSE: 236.1671
    Current temperature: 69.83, Best MSE: 236.1671
    Current temperature: 66.34, Best MSE: 236.1671
    Current temperature: 63.02, Best MSE: 236.1671
    Current temperature: 59.87, Best MSE: 236.1671
    Current temperature: 56.88, Best MSE: 236.1671
    Current temperature: 54.04, Best MSE: 236.1671
    Current temperature: 51.33, Best MSE: 236.1671
    Current temperature: 48.77, Best MSE: 235.2617
    Current temperature: 46.33, Best MSE: 235.2617
    Current temperature: 44.01, Best MSE: 233.7781
    Current temperature: 41.81, Best MSE: 233.7781
    Current temperature: 39.72, Best MSE: 233.7781
    Current temperature: 37.74, Best MSE: 233.7781
    Current temperature: 35.85, Best MSE: 233.7781
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 32.35, Best MSE: 233.7781
    Current temperature: 30.74, Best MSE: 233.7781
    Current temperature: 29.20, Best MSE: 226.3478
    Current temperature: 27.74, Best MSE: 226.3478
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 32.35, Best MSE: 233.7781
    Current temperature: 30.74, Best MSE: 233.7781
    Current temperature: 29.20, Best MSE: 226.3478
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 32.35, Best MSE: 233.7781
    Current temperature: 30.74, Best MSE: 233.7781
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 32.35, Best MSE: 233.7781
    Current temperature: 30.74, Best MSE: 233.7781
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 32.35, Best MSE: 233.7781
    Current temperature: 30.74, Best MSE: 233.7781
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 32.35, Best MSE: 233.7781
    Current temperature: 30.74, Best MSE: 233.7781
    Current temperature: 29.20, Best MSE: 226.3478
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 32.35, Best MSE: 233.7781
    Current temperature: 30.74, Best MSE: 233.7781
    Current temperature: 29.20, Best MSE: 226.3478
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 32.35, Best MSE: 233.7781
    Current temperature: 30.74, Best MSE: 233.7781
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 34.06, Best MSE: 233.7781
    Current temperature: 32.35, Best MSE: 233.7781
    Current temperature: 30.74, Best MSE: 233.7781
    Current temperature: 29.20, Best MSE: 226.3478
    Current temperature: 27.74, Best MSE: 226.3478
    Current temperature: 26.35, Best MSE: 226.3478
    Current temperature: 25.03, Best MSE: 226.3478
    Current temperature: 23.78, Best MSE: 226.3478
    Current temperature: 22.59, Best MSE: 226.3478
    Current temperature: 21.46, Best MSE: 226.3478
    Current temperature: 20.39, Best MSE: 226.3478
    Current temperature: 19.37, Best MSE: 226.3478
    Current temperature: 18.40, Best MSE: 226.3478
    Current temperature: 17.48, Best MSE: 226.3478
    Current temperature: 16.61, Best MSE: 225.6964
    Current temperature: 15.78, Best MSE: 225.6964
    Current temperature: 14.99, Best MSE: 225.6964
    Current temperature: 14.24, Best MSE: 225.6964
    Current temperature: 13.53, Best MSE: 225.6964
    Current temperature: 12.85, Best MSE: 225.6964
    Current temperature: 12.21, Best MSE: 225.6964
    Current temperature: 11.60, Best MSE: 225.6964
    Current temperature: 11.02, Best MSE: 222.2433
    Current temperature: 10.47, Best MSE: 218.6652
    Current temperature: 9.94, Best MSE: 218.6652
    Current temperature: 9.45, Best MSE: 211.6052
    Current temperature: 8.97, Best MSE: 211.6052
    Current temperature: 8.53, Best MSE: 211.6052
    Current temperature: 8.10, Best MSE: 211.6052
    Current temperature: 7.69, Best MSE: 211.6052
    Current temperature: 7.31, Best MSE: 192.1437
    Current temperature: 6.94, Best MSE: 192.1437
    Current temperature: 6.60, Best MSE: 192.1437
    Current temperature: 6.27, Best MSE: 192.1437
    Current temperature: 5.95, Best MSE: 192.1437
    Current temperature: 5.66, Best MSE: 192.1437
    Current temperature: 5.37, Best MSE: 192.1437
    Current temperature: 5.10, Best MSE: 192.1437
    Current temperature: 4.85, Best MSE: 192.1437
    Current temperature: 4.61, Best MSE: 192.1437
    Current temperature: 4.38, Best MSE: 192.1437
    Current temperature: 4.16, Best MSE: 192.1437
    Current temperature: 3.95, Best MSE: 192.1437
    Current temperature: 3.75, Best MSE: 192.1437
    Current temperature: 3.56, Best MSE: 192.1437
    Current temperature: 3.39, Best MSE: 192.1437
    Current temperature: 3.22, Best MSE: 192.1437
    Current temperature: 3.06, Best MSE: 192.1437
    Current temperature: 2.90, Best MSE: 192.1437
    Current temperature: 2.76, Best MSE: 192.1437
    Current temperature: 2.62, Best MSE: 192.1437
    Current temperature: 2.49, Best MSE: 192.1437
    Current temperature: 2.36, Best MSE: 192.1437
    Current temperature: 2.25, Best MSE: 192.1437
    Current temperature: 2.13, Best MSE: 192.1437
    Current temperature: 2.03, Best MSE: 192.1437
    Current temperature: 1.93, Best MSE: 192.1437
    Current temperature: 1.83, Best MSE: 192.1437
    Current temperature: 1.74, Best MSE: 192.1437
    Current temperature: 1.65, Best MSE: 192.1437
    Current temperature: 1.57, Best MSE: 192.1437
    Current temperature: 1.49, Best MSE: 192.1437
    Current temperature: 1.42, Best MSE: 192.1437
    Current temperature: 1.35, Best MSE: 192.1437
    Current temperature: 1.28, Best MSE: 192.1437
    Current temperature: 1.21, Best MSE: 192.1437
    Current temperature: 1.15, Best MSE: 192.1437
    Current temperature: 1.10, Best MSE: 192.1437
    Current temperature: 1.04, Best MSE: 192.1437
    Current temperature: 0.99, Best MSE: 192.1437""",
    """Current temperature: 95.00, Best MSE: 2.9065
    Current temperature: 90.25, Best MSE: 2.9065
    Current temperature: 85.74, Best MSE: 2.9065
    Current temperature: 81.45, Best MSE: 2.9065
    Current temperature: 77.38, Best MSE: 2.9065
    Current temperature: 73.51, Best MSE: 2.9065
    Current temperature: 69.83, Best MSE: 2.9065
    Current temperature: 66.34, Best MSE: 2.9065
    Current temperature: 63.02, Best MSE: 2.9065
    Current temperature: 59.87, Best MSE: 2.9065
    Current temperature: 56.88, Best MSE: 2.8398
    Current temperature: 54.04, Best MSE: 2.8398
    Current temperature: 51.33, Best MSE: 2.8398
    Current temperature: 48.77, Best MSE: 2.8398
    Current temperature: 46.33, Best MSE: 2.8398
    Current temperature: 44.01, Best MSE: 2.8398
    Current temperature: 41.81, Best MSE: 2.8398
    Current temperature: 39.72, Best MSE: 2.8398
    Current temperature: 37.74, Best MSE: 2.8398
    Current temperature: 35.85, Best MSE: 2.8398
    Current temperature: 34.06, Best MSE: 2.8398
    Current temperature: 32.35, Best MSE: 2.8398
    Current temperature: 30.74, Best MSE: 2.8398
    Current temperature: 29.20, Best MSE: 2.8398
    Current temperature: 27.74, Best MSE: 2.8398
    Current temperature: 26.35, Best MSE: 2.8398
    Current temperature: 25.03, Best MSE: 2.8398
    Current temperature: 23.78, Best MSE: 2.8398
    Current temperature: 22.59, Best MSE: 2.8398
    Current temperature: 21.46, Best MSE: 2.8398
    Current temperature: 20.39, Best MSE: 2.8398
    Current temperature: 19.37, Best MSE: 2.8398
    Current temperature: 18.40, Best MSE: 2.8398
    Current temperature: 17.48, Best MSE: 2.8398
    Current temperature: 16.61, Best MSE: 2.8398
    Current temperature: 15.78, Best MSE: 2.8398
    Current temperature: 14.99, Best MSE: 2.8398
    Current temperature: 14.24, Best MSE: 2.8398
    Current temperature: 13.53, Best MSE: 2.8398
    Current temperature: 12.85, Best MSE: 2.8398
    Current temperature: 12.21, Best MSE: 2.8398
    Current temperature: 11.60, Best MSE: 2.8398
    Current temperature: 11.02, Best MSE: 2.8398
    Current temperature: 10.47, Best MSE: 2.8398
    Current temperature: 9.94, Best MSE: 2.8398
    Current temperature: 9.45, Best MSE: 2.8398
    Current temperature: 8.97, Best MSE: 2.8398
    Current temperature: 8.53, Best MSE: 2.8398
    Current temperature: 8.10, Best MSE: 2.8398
    Current temperature: 7.69, Best MSE: 2.8398
    Current temperature: 7.31, Best MSE: 2.8398
    Current temperature: 6.94, Best MSE: 2.8398
    Current temperature: 6.60, Best MSE: 2.8398
    Current temperature: 6.27, Best MSE: 2.8398
    Current temperature: 5.95, Best MSE: 2.7419
    Current temperature: 5.66, Best MSE: 2.7195
    Current temperature: 5.37, Best MSE: 2.7195
    Current temperature: 5.10, Best MSE: 2.7195
    Current temperature: 4.85, Best MSE: 2.7195
    Current temperature: 4.61, Best MSE: 2.7195
    Current temperature: 4.38, Best MSE: 2.7195
    Current temperature: 4.16, Best MSE: 2.7195
    Current temperature: 3.95, Best MSE: 2.7195
    Current temperature: 3.75, Best MSE: 2.7195
    Current temperature: 3.56, Best MSE: 2.7195
    Current temperature: 3.39, Best MSE: 2.7195
    Current temperature: 3.22, Best MSE: 2.7195
    Current temperature: 3.06, Best MSE: 2.7195
    Current temperature: 2.90, Best MSE: 2.7195
    Current temperature: 2.76, Best MSE: 2.7195
    Current temperature: 2.62, Best MSE: 2.7195
    Current temperature: 2.49, Best MSE: 2.7195
    Current temperature: 2.36, Best MSE: 2.7195
    Current temperature: 2.25, Best MSE: 2.7195
    Current temperature: 2.13, Best MSE: 2.7195
    Current temperature: 2.03, Best MSE: 2.7195
    Current temperature: 1.93, Best MSE: 2.7195
    Current temperature: 1.83, Best MSE: 2.7195
    Current temperature: 1.74, Best MSE: 2.7195
    Current temperature: 1.65, Best MSE: 2.7195
    Current temperature: 1.57, Best MSE: 2.7195
    Current temperature: 1.49, Best MSE: 2.7195
    Current temperature: 1.42, Best MSE: 2.7195
    Current temperature: 1.35, Best MSE: 2.7195
    Current temperature: 1.28, Best MSE: 2.7195
    Current temperature: 1.21, Best MSE: 2.7195
    Current temperature: 1.15, Best MSE: 2.7195
    Current temperature: 1.10, Best MSE: 2.7195
    Current temperature: 1.04, Best MSE: 2.7195
    Current temperature: 0.99, Best MSE: 2.7195"""
]

# 针对每个标签进行基准实验并绘制图像
for label_idx, print_text in zip(label_columns, print_texts):
    baseline_mses = baseline_mses_list.get(label_idx, [])  # 获取对应的基准线列表
    temperatures, mses = parse_results_from_text(print_text)

    # 打印解析结果的长度
    print(f"Label {label_idx + 1}: 解析的温度数量 = {len(temperatures)}, MSE 数量 = {len(mses)}")

    plot_results_with_baseline(temperatures, mses, baseline_mses, label_idx)
