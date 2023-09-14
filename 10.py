list = input()
i = 0
flat = 0
while flat == 0 and i < len(list) - 1:
    if list[i] == list[i+1]:
        flat = 1
    else:
        i = i + 1
if flat == 1:
    print("Yes")
if flat == 0:
    print("No")
