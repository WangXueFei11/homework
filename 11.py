list = input()
list1 = []
for i in range(0,len(list)):
    list1.append(list[i])
for i in range(0,len(list1)):
    print(list1[len(list) - i - 1],end = "")
print(' ')

i =  len(list1) - 1
while i >= 0:
    print(list1[i],end = "")
    i = i - 1
