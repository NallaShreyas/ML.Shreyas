
# A7  Hyper 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def load_data(file_path, sheet_name='Sheet1'):
    """Load the dataset from an Excel file."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def preprocess_data(data, feature_1_name='signal', feature_2_name='rank'):
    """Extract features and create binary labels."""
    feature_1 = data[feature_1_name]
    feature_2 = data[feature_2_name]
    X = pd.DataFrame({'signal': feature_1, 'rank': feature_2})
    y = (feature_2 > feature_2.median()).astype(int)
    return X, y

def perform_grid_search(X_train, y_train):
    """Perform grid search to find the best k for kNN."""
    param_grid = {
        'n_neighbors': list(range(1, 21))  # Trying k values from 1 to 20
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(best_knn, X_test, y_test):
    """Evaluate the best model on the test set."""
    y_test_pred = best_knn.predict(X_test)
    print("\nTest set classification report:")
    print(classification_report(y_test, y_test_pred))
    print(f"Test set accuracy: {accuracy_score(y_test, y_test_pred):.2f}")

def generate_test_grid(feature_1_range, feature_2_range, step=0.1):
    """Generate a grid of test data points."""
    feature_1_test_values = np.arange(feature_1_range[0], feature_1_range[1] + step, step)
    feature_2_test_values = np.arange(feature_2_range[0], feature_2_range[1] + step, step)
    feature_1_grid, feature_2_grid = np.meshgrid(feature_1_test_values, feature_2_test_values)
    feature_1_flat = feature_1_grid.flatten()
    feature_2_flat = feature_2_grid.flatten()
    return pd.DataFrame({'signal': feature_1_flat, 'rank': feature_2_flat})

def plot_classification(X_test_grid, y_test_pred_grid, optimal_k):
    """Plot the test data points with predicted class colors."""
    colors = np.where(y_test_pred_grid == 0, 'blue', 'red')
    plt.figure(figsize=(10, 8))
    plt.scatter(X_test_grid['signal'], X_test_grid['rank'], c=colors, s=1, edgecolor='none')
    plt.title(f'kNN Classification with Optimal k={optimal_k}')
    plt.xlabel('Signal')
    plt.ylabel('Rank')
    plt.grid(True)
    plt.show()

def main(file_path):
    """Main function to execute the steps."""
    # Load and preprocess data
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Perform grid search for the best k
    grid_search = perform_grid_search(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # Evaluate the best model
    best_knn = grid_search.best_estimator_
    evaluate_model(best_knn, X_test, y_test)

    # Generate and classify the test grid
    X_test_grid = generate_test_grid(feature_1_range=(0, 10), feature_2_range=(0, 10))
    y_test_pred_grid = best_knn.predict(X_test_grid)

    # Plot classification results
    plot_classification(X_test_grid, y_test_pred_grid, optimal_k=grid_search.best_params_["n_neighbors"])

# Execute the main function
file_path = r"C:\\Users\\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"
main(file_path)
