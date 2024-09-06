#a4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data(file_path):
    """Load dataset from an Excel file."""
    return pd.read_excel(file_path, sheet_name='Sheet1')

def preprocess_data(data, signal_col='signal', rank_col='rank'):
    """Extract features and create binary labels."""
    signal_data = data[signal_col]
    rank_data = data[rank_col]
    
    # Combine the features into a single DataFrame
    X = pd.DataFrame({signal_col: signal_data, rank_col: rank_data})
    
    # Create binary labels
    y = (rank_data > rank_data.median()).astype(int)
    
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    """Split the dataset into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_test_grid(start=0, end=10, step=0.1):
    """Generate a grid of test data points."""
    values = np.arange(start, end + step, step)
    grid = np.meshgrid(values, values)
    
    # Flatten the grid to create test points
    flat_signal = grid[0].flatten()
    flat_rank = grid[1].flatten()
    
    return pd.DataFrame({'signal': flat_signal, 'rank': flat_rank})

def train_knn(X_train, y_train, k=3):
    """Train the kNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def classify_test_data(knn, X_test_grid):
    """Classify the test data using the trained kNN classifier."""
    y_test_pred = knn.predict(X_test_grid)
    X_test_grid['predicted_class'] = y_test_pred
    return X_test_grid

def plot_classification(X_test_grid):
    """Plot the test data with predicted class colors."""
    colors = np.where(X_test_grid['predicted_class'] == 0, 'blue', 'red')

    plt.figure(figsize=(10, 8))
    plt.scatter(X_test_grid['signal'], X_test_grid['rank'], c=colors, s=1, edgecolor='none')
    plt.title('Scatter Plot of Test Data Points Classified by kNN (k=3)')
    plt.xlabel('Signal')
    plt.ylabel('Rank')
    plt.grid(True)
    plt.show()

def main():
    file_path = r"C:\\Users\\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"
    
    # Load and preprocess data
    data = load_data(file_path)
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Create test grid
    X_test_grid = create_test_grid()
    
    # Train kNN classifier
    knn = train_knn(X_train, y_train)
    
    # Classify the test grid
    X_test_grid = classify_test_data(knn, X_test_grid)
    
    # Plot the classification results
    plot_classification(X_test_grid)

if __name__ == "__main__":
    main()
