list = [1,2,3,4,5]
for i in range(0,len(list)):
    print(list[len(list) - i - 1],end = " ")
print(' ')

i =  len(list) - 1
while i >= 0:
    print(list[i],end = " ")
    i = i - 1
