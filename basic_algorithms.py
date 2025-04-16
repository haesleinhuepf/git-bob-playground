def faculty(n):
    """
    Calculate the factorial of a number using recursion.

    Parameters
    ----------
    n : int
        Non-negative integer to calculate the factorial of.

    Returns
    -------
    int
        The factorial of the input number.
    """
    if n == 0 or n == 1:
        return 1
    else:
        return n * faculty(n - 1)

def bubble_sort(arr):
    """
    Sort an array of numbers using the bubble sort algorithm.

    Parameters
    ----------
    arr : list
        List of numbers to sort.

    Returns
    -------
    list
        Sorted list of numbers.
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def quick_sort(arr):
    """
    Sort an array of numbers using the quick sort algorithm.

    Parameters
    ----------
    arr : list
        List of numbers to sort.

    Returns
    -------
    list
        Sorted list of numbers.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
