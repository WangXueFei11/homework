x = int(input())
flat = 1
if x == 1:
    print('no')
elif x == 2:
    print('yes')
else:
    for i in range(2,int((x + 1) / 2) + 1):
        if x % i == 0:
            flat = 0
            break
    if flat == 0:
        print("no")
    else:
        print("yes")
