import random
import math
def estimate(times):
    sum = 0
    for i in range(times):
        x = random.uniform(2,3)
        y = random.uniform(0,100)
        d = x * x + 4 * x * math.sin(x) - y
        if d > 0:
            sum += 1
    print((sum * 100) / times)
estimate(1000000)
