c = int(input())
g = c / 2
i = 0
while abs(g * g * g - c) > 0.00000000001:
    g = (2 * g + c / (g * g)) / 3
    i += 1
    print("%d: %.15f" %(i,g))
