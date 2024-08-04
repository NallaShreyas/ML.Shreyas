import pandas as pd
import numpy as np

# Use forward slashes in the file path
s = "C:\Users\shrey\Downloads\Lab Session Data.xlsx"

# Load the correct sheet name
data = pd.read_excel(s, sheet_name='Purchase data')

# Extract relevant columns for matrix A
A = data[['Candies (#)', 'Mangoes (Kg)','Milk Packets (#)']].dropna()

# Extract relevant columns for matrix C
C=data[['Payment (Rs)']].dropna()

# Convert to NumPy array
A_matrix = A.to_numpy()

# Convert to NumPy array
C_MAtrix=C.to_numpy()


inverse_A = np.linalg.pinv(A_matrix)
X_matrix = np.dot(inverse_A, C_MAtrix)
print(X_matrix)