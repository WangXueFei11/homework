x = float(input())
list1 = []
i = 0
while x != 0:
    if x - 0.5 ** (i + 1) >= 0:
        x = x - 0.5 ** (i + 1)
        list1.append(1)
    else:
        list1.append(0)
    i += 1
print('0.',end = '')
for i in range(len(list1)):
    print(list1[i],end = '')
