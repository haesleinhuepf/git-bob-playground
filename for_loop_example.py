def print_odd_numbers():
    """
    Print odd numbers between 10 and 20.

    This function iterates through numbers 10 to 20 and prints only the odd numbers.
    """
    for i in range(10, 21):
        if i % 2 != 0:
            print(i)

if __name__ == "__main__":
    print_odd_numbers()
