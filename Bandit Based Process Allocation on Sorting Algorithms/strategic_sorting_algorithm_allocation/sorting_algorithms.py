import numpy as np

def selection_sort(arr, *args):
    n = len(arr)
    for i in range(n-1):
        min = arr[i]
        min_pos = i
        for j in range(i, n):
            if arr[j] < min:
                min = arr[j]
                min_pos = j
        arr[min_pos] = arr[i]
        arr[i] = min
    return arr

def bubble_sort(arr, *args):
    n = len(arr)
    for i in range(n-1):
        for j in range(n-1):
            if arr[j] > arr[j+1]:
                temp = arr[j+1]
                arr[j+1] = arr[j]
                arr[j] = temp
    return arr

def insertion_sort(arr, *args):
    n = len(arr)
    for i in range(1, n):
        current = arr[i]
        j = i-1
        while j>=0 and arr[j] > current:
            arr[j+1] = arr[j]
            j-=1
        arr[j+1] = current
    return arr

def merge_sort(arr, low, high):   
    n = high-low+1
    if n==1:
        return arr[low:high+1]
    else:
        mid = int((low+high)/2)
        left, right = merge_sort(arr, low, mid), merge_sort(arr, mid+1, high)
        # recombination
        return_arr = []
        i, j = 0, 0
        while i<len(left) and j<len(right):
            if left[i] < right[j]:
                return_arr.append(left[i])
                i += 1
            else:
                return_arr.append(right[j])
                j += 1
        if i==len(left):
            while j < len(right):
                return_arr.append(right[j])
                j += 1
        if j==len(right):
            while i < len(left):
                return_arr.append(left[i])
                i += 1
        return np.array(return_arr)

def quick_sort(arr, low, high):
    if low>=high:
        return
    i = low-1
    j = low
    pivot = arr[high]
    while j < high:
        if arr[j] < pivot:
            i += 1
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp
        j += 1
    temp = arr[i+1]
    arr[i+1] = pivot
    arr[high] = temp
    quick_sort(arr, low, i)
    quick_sort(arr, i+2, high)
    return arr

# Testing
# arr = np.array([7, 6, 5, 4, 3, 2, 1])

# print(selection_sort(arr))    
# print(bubble_sort(arr))
# print(insertion_sort(arr))
# print(merge_sort(arr, 0, len(arr)-1))
# print(quick_sort(arr, 0, len(arr)-1))