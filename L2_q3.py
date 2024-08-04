import pandas as pd
import numpy as np

s = "C:\Users\shrey\Downloads\Lab Session Data.xlsx"

# Load the correct sheet name
data = pd.read_excel(s, sheet_name='Purchase data')
print(data)
# Extract relevant columns for matrix A
cus = data[['Customer']].dropna()
customer=cus.to_numpy()

pur = data[['Payment (Rs)']].dropna()
purchase=pur.to_numpy()
print(purchase)
# convert from array to list as list can store different data types in a same list
customer=customer.tolist()
purchase=purchase.tolist()

'''storing customer data and their respective purchase data into a dictionary with key as the customer id 
and its value as the cost of the item and then appending either rich or poor into the value which is a list 
and then printing out rich and poor customers'''
dict={}
for i,j in enumerate(customer):
    j=str(j[0])

    dict[j]=purchase[i]
print(dict)

for i,j in dict.items():
    if j[0]>200:
        j.append("Rich")
    else:
        j.append("poor")
print(dict)

for j in dict:
    if dict[j][1] == "Rich":
        print(f"customer {j} is rich")
print()
for j in dict:
    if dict[j][1] == "poor":
        print(f"customer {j} is poor")






