import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Feature selection functions

def select_k_best_features(X_train, y_train, X_test, k=10):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Initialize a boolean mask to store selected feature indices
    selected_features_mask = np.zeros(X_train_flat.shape[1], dtype=bool)
    
    # Perform feature selection for each output column
    for i in range(y_train.shape[1]):  # Iterate over each output dimension
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_train_flat, y_train[:, i])
        selected_features_mask |= selector.get_support()  # Combine selected features into the mask
    
    # Select features based on the mask
    X_train_new = X_train_flat[:, selected_features_mask]
    X_test_new = X_test_flat[:, selected_features_mask]
    
    return X_train_new, X_test_new

def apply_pca(X_train, X_test, n_components=0.95):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    pca = PCA(n_components=n_components)
    X_train_new = pca.fit_transform(X_train_flat)
    X_test_new = pca.transform(X_test_flat)
    
    return X_train_new, X_test_new

def apply_lda(X_train, y_train, X_test, n_components=2):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_new = lda.fit_transform(X_train_flat, y_train[:, 0])  # Use the first column of y for supervision
    X_test_new = lda.transform(X_test_flat)
    
    return X_train_new, X_test_new

def apply_ica(X_train, X_test, n_components=10):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    ica = FastICA(n_components=n_components)
    X_train_new = ica.fit_transform(X_train_flat)
    X_test_new = ica.transform(X_test_flat)
    
    return X_train_new, X_test_new

def apply_rfe(X_train, y_train, X_test, model=None, n_features_to_select=10):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    if model is None:
        model = LinearRegression()
    
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    X_train_new = rfe.fit_transform(X_train_flat, y_train[:, 0])  # Use the first column of y for supervision
    X_test_new = rfe.transform(X_test_flat)
    
    return X_train_new, X_test_new

# Data preprocessing function with feature selection

def preprocess_data_with_selection(input_file='input_attributes.csv', labels_file='labels.csv', k=10):
    # Load data and process as before
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

    methods = ['kbest', 'pca', 'ica', 'lda', 'rfe', 'none']
    results = []

    for method in methods:
        if method == 'kbest':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train_selected, X_test_selected = select_k_best_features(X_train, y_train, X_test, k)
            method_name = 'SelectKBest'
        elif method == 'pca':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train_selected, X_test_selected = apply_pca(X_train, X_test, n_components=0.95)
            method_name = 'PCA'
        elif method == 'ica':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train_selected, X_test_selected = apply_ica(X_train, X_test, n_components=10)
            method_name = 'ICA'
        elif method == 'lda':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train_selected, X_test_selected = apply_lda(X_train, y_train, X_test, n_components=2)
            method_name = 'LDA'
        elif method == 'rfe':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LinearRegression()
            X_train_selected, X_test_selected = apply_rfe(X_train, y_train, X_test, model, n_features_to_select=10)
            method_name = 'RFE'
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train_selected = X_train.reshape(X_train.shape[0], -1)
            X_test_selected = X_test.reshape(X_test.shape[0], -1)
            method_name = 'No Feature Selection'
        
        results.append((X_train_selected, X_test_selected, y_train, y_test, method_name))
    
    return results

# Calculate metrics function
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    epsilon = 1e-10
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    mape = np.mean(np.abs((y_true_safe - y_pred) / y_true_safe), axis=0) * 100
    return mse, mae, rmse, r2, mape

# Training model with feature selection and evaluation

def train_model_with_feature_selection(X_train, X_test, y_train, y_test, model=None, method_name='Feature Selection'):
    if isinstance(model, MultiOutputRegressor):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    else:
        y_train_preds = np.zeros_like(y_train)
        y_test_preds = np.zeros_like(y_test)
        for i in range(y_train.shape[1]):
            model.fit(X_train, y_train[:, i])
            y_train_preds[:, i] = model.predict(X_train)
            y_test_preds[:, i] = model.predict(X_test)
        y_train_pred, y_test_pred = y_train_preds, y_test_preds

    train_mse, train_mae, train_rmse, train_r2, train_mape = calculate_metrics(y_train, y_train_pred)
    test_mse, test_mae, test_rmse, test_r2, test_mape = calculate_metrics(y_test, y_test_pred)

    print(f"Training Results ({method_name}):")
    print(f"MSE: {train_mse}\nMAE: {train_mae}\nRMSE: {train_rmse}\nR2: {train_r2}\nMAPE: {train_mape}")
    print(f"Testing Results ({method_name}):")
    print(f"MSE: {test_mse}\nMAE: {test_mae}\nRMSE: {test_rmse}\nR2: {test_r2}\nMAPE: {test_mape}")

    return train_mse, train_mae, train_rmse, train_r2, train_mape, test_mse, test_mae, test_rmse, test_r2, test_mape

# Main script
input_file = 'input_attributes.csv'
labels_file = 'labels.csv'
results = preprocess_data_with_selection(input_file, labels_file, k=20)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10, random_state=42))
for X_train, X_test, y_train, y_test, method_name in results:
    train_model_with_feature_selection(X_train, X_test, y_train, y_test, model, method_name)
