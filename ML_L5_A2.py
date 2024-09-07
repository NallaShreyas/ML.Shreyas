#a2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data(file_path, sheet_name='Sheet1'):
    """Load dataset from an Excel file."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def prepare_data(data, feature_col, target_col):
    """Prepare feature and target data."""
    X = data[feature_col].values.reshape(-1, 1)
    y = data[target_col]
    return X, y

def train_linear_regression(X_train, y_train):
    """Train a linear regression model."""
    reg = LinearRegression().fit(X_train, y_train)
    return reg

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mape = (mean_absolute_error(y_true, y_pred) / y_true.mean()) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

def main():
    # File path and column names
    file_path = r"C:\\Users\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"
    feature_col = 'signal'
    target_col = 'rank'
    
    # Load data
    data = load_data(file_path)
    
    # Prepare data
    X, y = prepare_data(data, feature_col, target_col)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    reg = train_linear_regression(X_train, y_train)
    
    # Predict on the training and test sets
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    
    # Calculate metrics
    mse_train, rmse_train, mape_train, r2_train = calculate_metrics(y_train, y_train_pred)
    mse_test, rmse_test, mape_test, r2_test = calculate_metrics(y_test, y_test_pred)
    
    # Output metrics
    print("Training Set Metrics:")
    print(f"MSE: {mse_train}")
    print(f"RMSE: {rmse_train}")
    print(f"MAPE: {mape_train}")
    print(f"R²: {r2_train}\n")

    print("Test Set Metrics:")
    print(f"MSE: {mse_test}")
    print(f"RMSE: {rmse_test}")
    print(f"MAPE: {mape_test}")
    print(f"R²: {r2_test}")

if __name__ == "__main__":
    main()
