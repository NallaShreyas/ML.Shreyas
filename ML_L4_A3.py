import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data(file_path, sheet_name='Sheet1'):
    """Load the dataset from an Excel file."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def preprocess_data(data, feature_1_name='signal', feature_2_name='rank'):
    """Extract features and create binary labels."""
    feature_1 = data[feature_1_name]
    feature_2 = data[feature_2_name]
    X = pd.DataFrame({'feature_1': feature_1, 'feature_2': feature_2})
    y = (feature_2 > feature_2.median()).astype(int)
    return X, y

def plot_sample_data(X_sample, y_sample):
    """Generate and plot random sample data."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X_sample[:, 0], X_sample[:, 1], c=['blue' if label == 0 else 'red' for label in y_sample], edgecolor='k')
    plt.title('Scatter Plot of 20 Random Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

def main():
    """Main function to execute the steps."""
    file_path = r"C:\\Users\\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"

    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # A3: Generate and plot random sample data
    np.random.seed(42)
    X_sample = np.random.uniform(1, 10, (20, 2))
    y_sample = (X_sample[:, 1] > X_sample[:, 0]).astype(int)
    plot_sample_data(X_sample, y_sample)
# Execute the main function
main()