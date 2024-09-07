#A7
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\\Users\\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"
data = pd.read_excel(file_path, sheet_name='Sheet1')  # Adjust the sheet name if needed

# Extract the 'signal' and 'rank' feature vectors (ignoring the target variable)
X = data[['signal', 'rank']]  # Adjust column names if necessary

# Split the dataset into train and test sets (though for clustering, we typically use all data)
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Initialize a list to store the distortions (inertia)
distortions = []

# Calculate the distortions for different k values
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train)
    distortions.append(kmeans.inertia_)  # Inertia measures the within-cluster sum of squares

# Plot the elbow plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, 20), distortions, marker='o')
plt.title('Elbow Plot for Determining Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.grid(True)
plt.show()

