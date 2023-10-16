import random
num_lists = []
length = []
for i in range(4):
    the_length = random.randint(5,10)
    length.append(the_length)
length.sort()
for i in length:
    num_list = [random.randint(1,100) for j in range(i)]
    num_lists.append(num_list)
print(num_lists)

def bubble_sort(array):
    n = len(array)
    for i in range(n):
        for j in range(1, n-i):
            if  array[j-1] > array[j] :
                array[j-1],array[j] = array[j], array[j-1]
    return array

def select_sort(array):
    n = len(array)
    for i in range(0,n):
        min = i
        for j in range(i+1,n):
            if array[j] < array[min] :
                min = j
        array[min],array[i] = array[i],array[min]
    return array

def insert_sort(array):
	count = len(array)
	for i in range(1, count):
		key = i - 1
		mark = array[i]
		while key >= 0 and array[key] > mark:
			array[key+1] = array[key]
			key -= 1
		array[key+1] = mark
	return array


def quick_sort(ary):
    return qsort(ary, 0, len(ary) - 1)

def qsort(ary, start, end):
    if start < end:
        left = start
        right = end
        key = ary[start]
    else:
        return ary
    while left < right:
        while left < right and ary[right] >= key:
            right -= 1
        if left < right:
            ary[left] = ary[right]
            left += 1
        while left < right and ary[left] < key:
            left += 1
        if left < right:
            ary[right] = ary[left]
            right -= 1
    ary[left] = key

    qsort(ary, start, left - 1)
    qsort(ary, left + 1, end)
    return ary
    
def walk_array(array):
    for i in range(len(array)):
        print(array[i],end = ' ')
    print('\n',end = "")

for i in range(4):
    bubble_sort(num_lists[i])
print('bubble_sort:')
for i in range(4):
    walk_array(num_lists[i])
for i in range(4):
    select_sort(num_lists[i])
print("select_sort:")
for i in range(4):
    walk_array(num_lists[i])
for i in range(4):
    insert_sort(num_lists[i])
print("insert_sort:")
for i in range(4):
    walk_array(num_lists[i])
for i in range(4):
    quick_sort(num_lists[i])
print("quick_sort:")
for i in range(4):
    walk_array(num_lists[i])
