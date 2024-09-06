#a5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data(file_path, sheet_name='Sheet1'):
    """Load the dataset from an Excel file."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def preprocess_data(data):
    """Extract features and create binary labels."""
    signal_data = data['signal']
    rank_data = data['rank']
    X = pd.DataFrame({'signal': signal_data, 'rank': rank_data})
    y = (rank_data > rank_data.median()).astype(int)
    return X, y

def generate_test_grid(signal_range, rank_range, step=0.1):
    """Generate a grid of test data points."""
    signal_test_values = np.arange(signal_range[0], signal_range[1] + step, step)
    rank_test_values = np.arange(rank_range[0], rank_range[1] + step, step)
    signal_test_grid, rank_test_grid = np.meshgrid(signal_test_values, rank_test_values)
    signal_test_flat = signal_test_grid.flatten()
    rank_test_flat = rank_test_grid.flatten()
    return pd.DataFrame({'signal': signal_test_flat, 'rank': rank_test_flat})

def classify_and_plot(X_train, y_train, X_test_grid, k_values):
    """Train kNN classifiers with various k values and plot results."""
    plt.figure(figsize=(18, 12))
    for i, k in enumerate(k_values):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_test_pred = knn.predict(X_test_grid)
        colors = np.where(y_test_pred == 0, 'blue', 'red')
        plt.subplot(2, 3, i + 1)
        plt.scatter(X_test_grid['signal'], X_test_grid['rank'], c=colors, s=1, edgecolor='none')
        plt.title(f'k = {k}')
        plt.xlabel('Signal')
        plt.ylabel('Rank')
        plt.grid(True)
    plt.suptitle('Scatter Plot of Test Data Points Classified by kNN with Various k Values')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    """Main function to execute the steps."""
    file_path = r"C:\\Users\\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test_grid = generate_test_grid(signal_range=(0, 10), rank_range=(0, 10))
    k_values = [1, 3, 5, 7, 9, 11]
    classify_and_plot(X_train, y_train, X_test_grid, k_values)

# Execute the main function

main()
