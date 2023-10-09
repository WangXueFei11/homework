def insertion_sort(array):
    for i in range(len(array)):
        cur_index = i
        while array[cur_index-1] > array[cur_index] and cur_index-1 >= 0:
            array[cur_index], array[cur_index-1] = array[cur_index-1], array[cur_index]
            cur_index -= 1
    return array
 
if __name__ == '__main__':
    array = [4, 17, 50, 7, 9, 24, 27, 20, 15, 5]
    print(insertion_sort(array))
