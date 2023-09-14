def cube_root(x):
    flag = 1
    if x == 0:
        print("0")
        return
    if x == 1:
        print("1")
        return
    if x == -1:
        print("-1")
        return
    if x < 0:
        x = -x
        flag = 0
    i = 0
    while i * i * i <= x:
        i = i + 1
        if i * i * i == x:
            if flag == 1:
                print(i)
            else:
                i = -i
                print(i)
            return
    l = i - 1
    r = i
    i = (l + r) / 2
    while x - i * i * i > 0.00001 or i * i * i - x > 0.00001:
        if x - i * i * i > 0.00001:
            l = i
        if i * i * i - x > 0.00001:
            r = i
        i = (l + r) / 2
    if flag == 1:
        print(i)
    else:
        i = -i
        print(i)

x = int(input())
cube_root(x)
