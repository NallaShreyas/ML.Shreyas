print("Lab1 -- Q1")
list1 = [2,7,4,1,3,6]
print("Given list is",list1)
count = 0
for i in range(len(list1)):
    for j in range(len(list1)):
        if list1[i] + list1[j] == 10:
            count += 1
print("Number of pairs of elements with sum 10 :",count)