list2=[]
size = int(input("Enter the number of elements in list :"))
if size < 3:
    print("Range Determination not possible...")
else:
    for i in range(size):
        item = int(input("Enter element :"))
        list2.append(item)
    print("Your list is ",list2)
#finding min  & max element
mini = list2[0]
maxi = list2[0]
# use loop to traverse and check
for j in range(len(list2)):
    if list2[j] < mini:
        mini = list2[j]
    elif list2[j] > maxi:
        maxi = list2[j]
range = maxi - mini
print("Range of given list is:",range)
