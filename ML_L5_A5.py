#A5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_train)
    return kmeans

def calculate_clustering_metrics(X_train, kmeans):
    """Calculate clustering metrics."""
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_train, labels)
    ch_score = calinski_harabasz_score(X_train, labels)
    db_index = davies_bouldin_score(X_train, labels)
    return silhouette_avg, ch_score, db_index

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
    kmeans = perform_kmeans_clustering(X_train, n_clusters=2)
    
    # Calculate and output metrics
    silhouette_avg, ch_score, db_index = calculate_clustering_metrics(X_train, kmeans)
    
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")

if __name__ == "__main__":
    main()
