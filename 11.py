list = input()
list1 = []
for i in range(0,len(list)):
    if list[i] != " ":
        list1.append(list[i])
for i in range(0,len(list1)):
    print(list1[i],end = "")
print(" ")
