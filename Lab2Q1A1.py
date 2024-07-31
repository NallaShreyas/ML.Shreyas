import pandas as pd

file_path = 'C:\Users\shrey\Downloads\Lab Session Data.xlsx'
sheet_name = 'Grocery'
data = pd.read_excel(file_path,sheet_name=sheet_name)
#####
A = data[['Feature1', 'Feature2', 'FeatureN']].values
C = data[['Price']].values
#####
dimensionality = A.shape[1]
print(f"Dimensionality of vector space :{dimensionality}")
#########
num_vectors = A.shape[0]
printf(f"Number of vectors in vector space: {num_vectors}")
#####
import numpy as np
rank_A = np.linalg.matrix_rank(A)
print(f"Rank of amtrix A:{rank_A} ")
#####
A_pseudo_inv = np.linalg.pinv(A)

estimated_costs = A_pseudo_inv @ C
print(f"Estimated cost of each product : {estimated_costs}")