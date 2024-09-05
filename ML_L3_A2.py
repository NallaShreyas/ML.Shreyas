'''import pandas as pd
import numpy as np
import matplotlib.plot as plt

# Load the Excel file
file_path = r"C:\\Users\\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"  
data = pd.read_excel(file_path)

# Proceed with your data processing steps
# Example: Selecting the 'signal' class feature
signal_data = data['signal']  # Adjust the column name if 'signal' is named differently in your dataset

# Calculate the mean and variance of the 'signal' feature
signal_mean = np.mean(signal_data)
signal_variance = np.var(signal_data)

# Generate the histogram using matplotlib
plt.figure(figsize=(10, 6))
plt.hist(signal_data, bins=20, alpha=0.75, color='blue', edgecolor='black')
plt.title('Histogram of the Signal Feature')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# Add mean line and text
plt.axvline(signal_mean, color='red', linestyle='dashed', linewidth=1)
plt.text(signal_mean * 1.1, plt.ylim()[1] * 0.9, f'Mean: {signal_mean:.2f}', color='red')

# Display the histogram
plt.show()

# Print the calculated mean and variance
print(f"Mean of the 'signal' feature: {signal_mean:.2f}")
print(f"Variance of the 'signal' feature: {signal_variance:.2f}")'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the Excel file."""
    return pd.read_excel(file_path)

def select_feature(data, column_name):
    """Select a specific feature (column) from the dataset."""
    return data[column_name]

def calculate_statistics(data):
    """Calculate the mean and variance of the data."""
    mean = np.mean(data)
    variance = np.var(data)
    return mean, variance

def plot_histogram(data, mean, title, xlabel, ylabel, bins=20):
    """Generate and display a histogram of the data."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, alpha=0.75, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # Add mean line and text
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean * 1.1, plt.ylim()[1] * 0.9, f'Mean: {mean:.2f}', color='red')
    
    plt.show()

def main():
    # Load the Excel file
    file_path = r"C:\\Users\\shrey\\Downloads\\Feature Extraction using TF-IDF.xlsx"
    data = load_data(file_path)

    # Select the 'signal' feature
    signal_data = select_feature(data, 'signal')

    # Calculate statistics
    signal_mean, signal_variance = calculate_statistics(signal_data)

    # Plot the histogram
    plot_histogram(signal_data, signal_mean, 'Histogram of the Signal Feature', 'Value', 'Frequency')

    # Print the calculated mean and variance
    print(f"Mean of the 'signal' feature: {signal_mean:.2f}")
    print(f"Variance of the 'signal' feature: {signal_variance:.2f}")

main()

