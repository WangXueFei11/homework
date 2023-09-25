def max_multi(n):
    if n % 3 == 0:
        result = 3 ** (n / 3)
    elif n % 3 == 1:
        result = 4 * 3 ** ((n - 4) / 3)
    elif n % 3 == 2:
        result = 2 * 3 ** ((n - 2) / 3)
    print(result)

n = int(input())
max_multi(n)
