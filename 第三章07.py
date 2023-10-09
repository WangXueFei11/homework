x = int(input())
y = int(input())
MIN = min(x,y)
flag = 0
i = MIN
while i >= 1 and flag == 0:
    if(x % i == 0) and (y % i == 0):
        flag = 1
    i -= 1
if flag == 0:
    print("1")
else:
    print(i + 1)
