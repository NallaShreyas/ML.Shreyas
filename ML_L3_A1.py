import pandas as pd
import numpy as np

def load_data(file_path, sheet_name='Sheet1'):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def calculate_centroid(df, column_name):
    return np.mean(df[column_name])

def calculate_spread(df, column_name):
    return np.std(df[column_name])

def calculate_euclidean_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)

def main():
    file_path = r"C:\\Users\\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"
    data = load_data(file_path)

    df = data[['rank', 'signal']]

    rank_centroid = calculate_centroid(df, 'rank')
    signal_centroid = calculate_centroid(df, 'signal')

    rank_spread = calculate_spread(df, 'rank')
    signal_spread = calculate_spread(df, 'signal')

    distance = calculate_euclidean_distance(rank_centroid, signal_centroid)

    print(f"Rank Centroid: {rank_centroid}")
    print(f"Signal Centroid: {signal_centroid}")
    print(f"Rank Spread (Standard Deviation): {rank_spread}")
    print(f"Signal Spread (Standard Deviation): {signal_spread}")
    print(f"Distance between Centroids: {distance}")

main()