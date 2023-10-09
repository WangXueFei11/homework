B = []
list1 = input()
list1 = list1.split(' ')
sum = 1
for i in list1:
       sum *= int(i)
       B.append(sum)
print(B)
