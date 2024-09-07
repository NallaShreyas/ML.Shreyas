#a4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def load_data(file_path, sheet_name='Sheet1'):
    """Load dataset from an Excel file."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def prepare_features(data, feature_cols):
    """Prepare feature data for clustering."""
    X = data[feature_cols]
    return X

def perform_kmeans_clustering(X_train, n_clusters=2):
    """Perform K-Means clustering on the data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_train)
    return kmeans

def main():
    # File path and feature column names
    file_path = r"C:\\Users\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"
    feature_cols = ['signal', 'rank']
    
    # Load data
    data = load_data(file_path)
    
    # Prepare features
    X = prepare_features(data, feature_cols)
    
    # Split data
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
    
    # Perform K-Means clustering
    kmeans = perform_kmeans_clustering(X_train, n_clusters=2) # Here , k = 2
    
    # Get cluster labels and centers
    cluster_labels_train = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # Output results
    print("Cluster Labels for Training Data:")
    print(cluster_labels_train)

    print("\nCluster Centers:")
    print(cluster_centers)

if __name__ == "__main__":
    main()
