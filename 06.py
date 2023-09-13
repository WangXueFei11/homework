w = int(input())
x = int(input())
y = int(input())
z = int(input())
list = []
list.append(w)
list.append(x)
list.append(y)
list.append(z)
list.sort(reverse = True)
for i in list:
    print(i,end = " ")
