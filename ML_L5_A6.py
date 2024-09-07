#a6
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

def load_data(file_path, sheet_name='Sheet1'):
    """Load dataset from an Excel file."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def prepare_features(data, feature_cols):
    """Prepare feature data for clustering."""
    X = data[feature_cols]
    return X

def perform_kmeans_clustering(X_train, k):
    """Perform K-Means clustering on the data."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train)
    return kmeans

def calculate_clustering_metrics(X_train, kmeans):
    """Calculate clustering metrics."""
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_train, labels)
    ch_score = calinski_harabasz_score(X_train, labels)
    db_index = davies_bouldin_score(X_train, labels)
    return silhouette_avg, ch_score, db_index

def plot_clustering_metrics(k_values, silhouette_scores, ch_scores, db_indices):
    """Plot clustering metrics against k values."""
    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Score vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')

    plt.subplot(3, 1, 2)
    plt.plot(k_values, ch_scores, marker='o', color='green')
    plt.title('Calinski-Harabasz Score vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')

    plt.subplot(3, 1, 3)
    plt.plot(k_values, db_indices, marker='o', color='red')
    plt.title('Davies-Bouldin Index vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Index')

    plt.tight_layout()
    plt.show()

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
    
    # Initialize lists to store evaluation scores
    k_values = range(2, 11)
    silhouette_scores = []
    ch_scores = []
    db_indices = []
    
    # Perform clustering for each k value and calculate metrics
    for k in k_values:
        kmeans = perform_kmeans_clustering(X_train, k)
        silhouette_avg, ch_score, db_index = calculate_clustering_metrics(X_train, kmeans)
        silhouette_scores.append(silhouette_avg)
        ch_scores.append(ch_score)
        db_indices.append(db_index)
    
    # Plot metrics
    plot_clustering_metrics(k_values, silhouette_scores, ch_scores, db_indices)

if __name__ == "__main__":
    main()
