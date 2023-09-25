def square_root_3(x):
    c = x 
    g = c / 2
    i = 0
    while abs((g * g) - c) > 0.00000000001:
        g = (g + c / g) / 2
        i += 1
        print("%d:%.13f"%(i,g))
x = int(input())
square_root_3(x)
