import time
start_time = time.time()
for i in range(1000000):
    pass
end_time = time.time()
execution_time = end_time - start_time
print("start_time:",start_time)
print("end_time:",end_time)
print("execution_time:",execution_time)
