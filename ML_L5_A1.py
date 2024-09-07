#a1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

def evaluate_model(reg, X_train, y_train):
    """Evaluate the linear regression model."""
    y_train_pred = reg.predict(X_train)
    return {
        "coefficients": reg.coef_,
        "intercept": reg.intercept_,
        "predictions": y_train_pred[:5],
        "actual_values": y_train[:5].values
    }

def main():
    # File path and column names
    file_path = r"C:\\Users\\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"
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
    
    # Evaluate model
    results = evaluate_model(reg, X_train, y_train)
    
    # Output results
    print(f"Model Coefficient: {results['coefficients']}")
    print(f"Model Intercept: {results['intercept']}")
    print(f"First few predictions on the training set: {results['predictions']}")
    print(f"First few actual target values: {results['actual_values']}")

if __name__ == "__main__":
    main()
